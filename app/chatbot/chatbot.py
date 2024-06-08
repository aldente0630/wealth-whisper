import logging
import os
import sys
from typing import Dict, List, Optional, Union
import numpy as np
import tenacity
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
from langchain_community.callbacks import StreamlitCallbackHandler

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
)
from app.prompts import get_condensed_question_prompt, get_qa_prompt
from app.utils import (
    calc_cos_sim,
    ChatModelId,
    convert_date_string,
    get_embeddings,
    get_llm,
    get_memory,
    get_model_id,
    get_name,
    get_retriever,
    load_config,
    logger,
)


class ChatBot:
    def __init__(
        self,
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        domain_name: Optional[str] = None,
        index_name: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        secret_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        retriever_type: Optional[str] = None,
        k: int = 5,
        multiplier: int = 1,
        semantic_weight: float = 0.5,
        use_hyde: bool = False,
        use_rag_fusion: bool = False,
        use_parent_document: bool = False,
        use_time_weight: bool = False,
        use_reorder: bool = False,
        decay_rate: float = 0.01,
        temperature: float = 1.0,
        use_memory: bool = False,
        st_callback: Optional[StreamlitCallbackHandler] = None,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.profile_name = profile_name
        self.region_name = region_name
        self.domain_name = domain_name
        self.index_name = index_name
        self.endpoint_name = endpoint_name
        self.secret_name = secret_name
        self.embedding_model_id = get_model_id(embedding_model_name)
        self.retriever_type = retriever_type.lower()
        self.k = k
        self.multiplier = multiplier
        self.semantic_weight = semantic_weight
        self.use_hyde = use_hyde
        self.use_rag_fusion = use_rag_fusion
        self.use_parent_document = use_parent_document
        self.use_time_weight = use_time_weight
        self.use_reorder = use_reorder
        self.decay_rate = decay_rate
        self.temperature = temperature
        self.use_memory = use_memory
        self.st_callback = st_callback
        self.verbose = verbose
        self.logger = logger

        self.model_id = ChatModelId.CLAUDE_V3_SONNET  # You can change the chat model ID
        self.version = "0.0.1"

        self.condense_question_prompt = get_condensed_question_prompt()
        self.qa_prompt = get_qa_prompt()
        self.callbacks = [StdOutCallbackHandler()]
        self.retriever = get_retriever(
            self.profile_name,
            self.region_name,
            self.domain_name,
            self.index_name,
            self.endpoint_name,
            self.secret_name,
            self.embedding_model_id,
            self.retriever_type,
            self.k,
            self.multiplier,
            self.semantic_weight,
            self.use_hyde,
            self.use_rag_fusion,
            self.use_parent_document,
            self.use_time_weight,
            self.use_reorder,
            self.decay_rate,
            self.temperature,
            verbose=self.verbose,
            logger=self.logger,
        )
        self.memory = (
            get_memory("ConversationBufferWindowMemory") if use_memory else None
        )
        self.chat_history = []

        self.embeddings = get_embeddings(
            self.embedding_model_id,
            self.endpoint_name,
            profile_name=self.profile_name,
            region_name=self.region_name,
            logger=self.logger,
        )
        self.min_thr, self.max_thr = (
            0.5,
            0.75,
        )  # These values have been decided heuristically

    def batch(self, questions: List[str]) -> List[Dict[str, str]]:
        qa_chain = RetrievalQA.from_chain_type(
            get_llm(
                self.model_id,
                self.temperature,
                profile_name=self.profile_name,
                region_name=self.region_name,
            ),
            callbacks=self.callbacks,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.qa_prompt},
        )

        return qa_chain.batch(
            [{"query": question} for question in questions],
            config={"max_concurrency": 5},
        )

    @tenacity.retry(
        wait=tenacity.wait_fixed(10),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    def invoke(
        self,
        question: str,
        st_callback: Optional[StreamlitCallbackHandler] = None,
    ) -> Dict[str, Union[Optional[str], List[str]]]:
        qa_chain = ConversationalRetrievalChain.from_llm(
            get_llm(
                self.model_id,
                self.temperature,
                profile_name=self.profile_name,
                region_name=self.region_name,
                callbacks=None if st_callback is None else [st_callback],
                streaming=bool(st_callback),
            ),
            self.retriever,
            callbacks=self.callbacks,
            condense_question_prompt=self.condense_question_prompt,
            chain_type="stuff",
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.qa_prompt},
        )

        result = qa_chain.invoke(
            {"question": question, "chat_history": self.chat_history}
        )
        source_docs = result["source_documents"]

        if len(source_docs) > 0:
            emb_ans = np.array(self.embeddings.embed_query(result["answer"]))
            emb_docs = np.array(
                self.embeddings.embed_documents(
                    [source_doc.page_content for source_doc in source_docs]
                )
            )
            scores = np.array([calc_cos_sim(emb_ans, emb_doc) for emb_doc in emb_docs])

            if np.max(scores) > self.min_thr:
                indexes = np.arange(len(scores))[scores > self.max_thr].tolist()
                if len(indexes) == 0:
                    indexes = [np.argmax(scores)]

            else:
                indexes = []

            source_descriptions = []
            for i in indexes:
                base_date = source_docs[i].metadata.get("base_date", "")
                item = source_docs[i].metadata.get("item", "")
                publisher = source_docs[i].metadata.get("publisher", "")
                title = source_docs[i].metadata.get("title", "")

                title = f"{item} - {title}" if len(item) > 0 else title
                source_descriptions.append(
                    f"{publisher}: {title} ({convert_date_string(base_date)})"
                )

            source_descriptions = (
                ", ".join(set(source_descriptions))
                if len(source_descriptions) > 0
                else None
            )

        else:
            source_descriptions = None

        return {"answer": result["answer"], "source_descriptions": source_descriptions}


def get_dir_path(dir_name: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), dir_name))


if __name__ == "__main__":
    config_dir = get_dir_path(os.path.join(os.pardir, os.pardir, "app", "configs"))
    config = load_config(os.path.join(config_dir, "config.yaml"))

    chatbot = ChatBot(
        profile_name=config.profile_name,
        region_name=config.region_name,
        domain_name=get_name(config.proj_name, config.stage, "docs"),
        index_name=get_name(config.proj_name, config.stage, "docs"),
        endpoint_name=get_name(config.proj_name, config.stage, "embedding"),
        secret_name=get_name(config.proj_name, config.stage, "docs"),
        embedding_model_name=config.embedding_model_name,
        retriever_type=config.retriever_type,
        k=config.k,
        multiplier=config.multiplier,
        semantic_weight=config.semantic_weight,
        use_hyde=config.use_hyde,
        use_rag_fusion=config.use_rag_fusion,
        use_parent_document=config.use_parent_document,
        use_time_weight=config.use_time_weight,
        use_reorder=config.use_reorder,
        decay_rate=config.decay_rate,
        temperature=config.temperature,
        verbose=True,
        logger=logger,
    )

    # Add your test samples
    queries = ["삼성전자 예상 실적은?", "미국 금리 전망은?"]

    for query in queries:
        results = chatbot.invoke(query)
        string = "-" * 100 + f"\nquestion: {query}\nanswer: {results['answer']}"
        logger.info(string)
