import logging
import json
import math
import os
import sys
from datetime import datetime
from functools import partial
from multiprocessing.pool import ThreadPool
from operator import itemgetter
from pprint import pformat
from typing import Any, Dict, Final, Iterable, List, Optional, Tuple, Union
import boto3
import pandas as pd
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
)
from langchain.schema import BaseRetriever, Document
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_models import BedrockChat, ChatOpenAI
from langchain_community.document_transformers import LongContextReorder
from langchain_community.embeddings import (
    BedrockEmbeddings,
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
)
from opensearchpy import OpenSearch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from app.prompts.prompts import (
    get_hyde_prompt,
    get_rag_fusion_prompt,
    get_summary_prompt,
)
from app.utils.aws_helpers import (
    create_opensearch_client,
    EmbeddingsContentHandlerWrapper,
    get_opensearch_domain_endpoint,
    get_opensearch_query,
    get_secret,
    SagemakerEndpointEmbeddingsWrapper,
)
from app.utils.common import (
    ChatModelId,
    get_dimensions,
    get_provider_name,
)
from app.utils.misc import get_default, log_or_print


SEARCH_THREAD_POOL: Final = ThreadPool(processes=2)
TASK_THREAD_POOL: Final = ThreadPool(processes=5)


class BedrockEmbeddingsWrapper(BedrockEmbeddings):
    max_str_len: Optional[int] = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            # Truncates the text to the maximum length
            response = self._embedding_func(
                text if self.max_str_len is None else text[: self.max_str_len]
            )

            if self.normalize:
                response = self._normalize_vector(response)

            results.append(response)

        return results


class OpenSearchRetriever(BaseRetriever):
    opensearch_client: OpenSearch
    index_name: str
    embeddings: Optional[Embeddings] = None
    k: int = 5
    retriever_type: Optional[str] = None
    boolean_filter: Optional[List[Dict[str, Dict[str, str]]]] = None
    minimum_should_match: int = 0
    use_parent_document: bool = False
    reorder: Optional[LongContextReorder] = None
    verbose: bool = False
    logger: Optional[logging.Logger] = None

    def _get_parent_documents(
        self, child_search_results: List[Document]
    ) -> List[Document]:
        parent_ids = list(
            dict.fromkeys([doc.metadata["parent"] for doc in child_search_results])
        )

        responses = self.opensearch_client.mget(
            body={"ids": parent_ids}, index=self.index_name
        )
        parent_search_results = [
            Document(
                page_content=response["_source"]["text"],
                metadata=response["_source"]["metadata"],
            )
            for response in responses.get("docs", [])
            if response.get("_source")
        ]

        return parent_search_results

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        search_results = search_in_opensearch(
            self.opensearch_client,
            self.index_name,
            query,
            k=self.k,
            retriever_type=self.retriever_type,
            boolean_filter=self.boolean_filter,
            minimum_should_match=self.minimum_should_match,
            embeddings=self.embeddings,
            use_ensemble=False,
        )

        if self.use_parent_document:
            search_results = self._get_parent_documents(search_results)

        if self.reorder is not None:
            search_results = self.reorder.transform_documents(search_results)

        if self.verbose:
            log_or_print(
                f"Search results: {pformat(search_results)}",
                logger=self.logger,
            )

        return search_results


class OpenSearchEnsembleRetriever(BaseRetriever):
    opensearch_client: OpenSearch
    index_name: str
    embeddings: Embeddings
    llm: Optional[BaseChatModel] = None
    k: int = 5
    multiplier: int = 1
    boolean_filter: Optional[List[Dict[str, Dict[str, str]]]] = None
    minimum_should_match: int = 0
    query_augmentation_size: int = 3
    weights: Optional[List[float]] = None
    algorithm: Optional[str] = None
    c: int = 60
    profile_name: Optional[str] = None
    region_name: Optional[str] = None
    verbose: bool = False
    logger: Optional[logging.Logger] = None

    use_hyde: bool = False
    use_rag_fusion: bool = False
    use_parent_document: bool = False
    use_time_weight: bool = False
    reorder: Optional[LongContextReorder] = None
    decay_rate: float = 0.01
    is_async: bool = False

    algorithm = get_default(algorithm, "rrf")
    algorithms = ["rrf", "simple_weighted"]
    if algorithm.lower() not in algorithms:
        raise ValueError(f"Invalid algorithm: {algorithm}")

    if (use_hyde or use_rag_fusion) and llm is None:
        raise ValueError("LLM must be provided when using HyDE or RAG fusion.")

    if use_hyde and use_rag_fusion:
        raise ValueError("HyDE and RAG fusion cannot be used together.")

    weights = get_default(weights, [0.5, 0.5])

    region_name = get_default(region_name, "us-east-1")
    boto_session = boto3.Session(region_name=region_name, profile_name=profile_name)

    def _get_ensemble_results(
        self,
        docs_list: List[List[Tuple[Document, float]]],
        weights: List[float],
    ) -> List[Tuple[Document, float]]:
        # Create a union of all documents
        all_docs = set()
        for docs in docs_list:
            for doc, _ in docs:
                all_docs.add(doc.page_content)

        # Initialize each document's score
        scores = {doc: 0.0 for doc in all_docs}

        # Calculate RRF scores for each document
        for docs, weight in zip(docs_list, weights):
            for i, (doc, score) in enumerate(docs, start=1):
                if self.algorithm.lower() == "rrf":
                    score = weight * (1.0 / (i + self.c))
                else:
                    score *= weight
                scores[doc.page_content] += score

        # Sort documents by score in descending order
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Remaps the sorted page content back to the original document object
        page_content_to_doc_map = {
            doc.page_content: doc for docs in docs_list for (doc, _) in docs
        }

        sorted_docs = [
            (page_content_to_doc_map[page_content], score)
            for (page_content, score) in sorted_docs
        ]

        return sorted_docs[: self.k * self.multiplier]

    def _get_parent_documents(
        self, child_search_results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        metadata = {}
        for doc, score in child_search_results:
            parent_id = doc.metadata["parent"]
            if parent_id not in metadata:
                metadata[parent_id] = score

        parent_ids = list(metadata.keys())
        scores = list(metadata.values())

        responses = self.opensearch_client.mget(
            body={"ids": parent_ids}, index=self.index_name
        )
        parent_search_results = [
            (
                Document(
                    page_content=response["_source"]["text"],
                    metadata=response["_source"]["metadata"],
                ),
                score,
            )
            for response, score in zip(responses.get("docs", []), scores)
            if response.get("_source")
        ]

        return parent_search_results

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self._search_with_ensemble(query)

    def _get_time_weighted_results(
        self, docs: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        curr_date = datetime.now().date()

        time_weighted_results = []
        for doc, score in docs:
            prev_date = datetime.strptime(doc.metadata["base_date"], "%Y%m%d").date()
            discount_factor = math.exp(
                -self.decay_rate * (curr_date - prev_date).days / 365.25
            )
            time_weighted_results.append((doc, score * discount_factor))

        return sorted(time_weighted_results, key=lambda x: x[1], reverse=True)

    def _search_semantically_by_hyde(self, query: str):
        chain = (
            {"query": itemgetter("query")}
            | get_hyde_prompt()
            | self.llm
            | StrOutputParser()
        )

        hyde_responses = [chain.invoke({"query": query})]
        hyde_responses.insert(0, query)

        tasks = []
        for hyde_response in hyde_responses:
            semantic_search = partial(
                search_in_opensearch,
                self.opensearch_client,
                self.index_name,
                hyde_response,
                retriever_type="semantic",
                k=self.k * self.multiplier,
                boolean_filter=self.boolean_filter,
                embeddings=self.embeddings,
                use_ensemble=True,
            )
            tasks.append(
                TASK_THREAD_POOL.apply_async(
                    semantic_search,
                )
            )
        doc_lists = [task.get() for task in tasks]

        results = self._get_ensemble_results(
            doc_lists,
            [1.0 / len(doc_lists)] * len(doc_lists),
        )

        if self.verbose:
            log_or_print(
                f"Generated answers for HyDE: {pformat(hyde_responses)}",
                logger=self.logger,
            )

        return results

    def _search_semantically_by_rag_fusion(self, query: str):
        chain = (
            {
                "query": itemgetter("query"),
                "query_augmentation_size": itemgetter("query_augmentation_size"),
            }
            | get_rag_fusion_prompt()
            | self.llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        rag_fusion_queries = chain.invoke(
            {
                "query": query,
                "query_augmentation_size": self.query_augmentation_size,
            }
        )

        rag_fusion_queries = [query for query in rag_fusion_queries if query != ""]
        if len(rag_fusion_queries) > self.query_augmentation_size:
            rag_fusion_queries = rag_fusion_queries[-self.query_augmentation_size :]
        rag_fusion_queries.insert(0, query)

        tasks = []
        for rag_fusion_query in rag_fusion_queries:
            semantic_search = partial(
                search_in_opensearch,
                self.opensearch_client,
                self.index_name,
                rag_fusion_query,
                retriever_type="semantic",
                k=self.k * self.multiplier,
                boolean_filter=self.boolean_filter,
                embeddings=self.embeddings,
                use_ensemble=True,
            )
            tasks.append(
                TASK_THREAD_POOL.apply_async(
                    semantic_search,
                )
            )
        doc_lists = [task.get() for task in tasks]

        results = self._get_ensemble_results(
            doc_lists,
            [1.0 / (self.query_augmentation_size + 1.0)]
            * (self.query_augmentation_size + 1),
        )

        if self.verbose:
            log_or_print(
                f"Generated questions for RAG Fusion: {rag_fusion_queries}",
                logger=self.logger,
            )

        return results

    def _search_with_ensemble(
        self,
        query: str,
    ) -> List[Document]:
        if self.is_async:
            search_lexically = partial(
                search_in_opensearch,
                self.opensearch_client,
                self.index_name,
                query,
                k=self.k * self.multiplier,
                retriever_type="lexical",
                boolean_filter=self.boolean_filter,
                minimum_should_match=self.minimum_should_match,
                use_ensemble=True,
            )

            if self.use_hyde:
                search_semantically = partial(self._search_semantically_by_hyde, query)

            elif self.use_rag_fusion:
                search_semantically = partial(
                    self._search_semantically_by_rag_fusion, query
                )

            else:
                search_semantically = partial(
                    search_in_opensearch,
                    self.opensearch_client,
                    self.index_name,
                    query,
                    retriever_type="semantic",
                    k=self.k * self.multiplier,
                    boolean_filter=self.boolean_filter,
                    embeddings=self.embeddings,
                    use_ensemble=True,
                )

            lexical_search_threads = SEARCH_THREAD_POOL.apply_async(search_lexically)
            sematic_search_threads = SEARCH_THREAD_POOL.apply_async(search_semantically)

            lexical_search_results, semantic_search_results = (
                lexical_search_threads.get(),
                sematic_search_threads.get(),
            )

        else:
            lexical_search_results = search_in_opensearch(
                self.opensearch_client,
                self.index_name,
                query,
                retriever_type="lexical",
                k=self.k * self.multiplier,
                boolean_filter=self.boolean_filter,
                minimum_should_match=self.minimum_should_match,
                use_ensemble=True,
            )

            if self.use_hyde:
                semantic_search_results = self._search_semantically_by_hyde(query)

            elif self.use_rag_fusion:
                semantic_search_results = self._search_semantically_by_rag_fusion(query)

            else:
                semantic_search_results = search_in_opensearch(
                    self.opensearch_client,
                    self.index_name,
                    query,
                    retriever_type="semantic",
                    k=self.k * self.multiplier,
                    boolean_filter=self.boolean_filter,
                    embeddings=self.embeddings,
                    use_ensemble=True,
                )

        if self.verbose:
            log_or_print(
                f"""Lexical search results: {pformat(lexical_search_results)}\n
                Semantic search results: {pformat(semantic_search_results)}""",
                logger=self.logger,
            )

        ensemble_search_results = self._get_ensemble_results(
            [lexical_search_results, semantic_search_results],
            self.weights,
        )

        if self.use_time_weight:
            ensemble_search_results = self._get_time_weighted_results(
                ensemble_search_results
            )

        if self.use_parent_document:
            ensemble_search_results = self._get_parent_documents(
                ensemble_search_results
            )

        ensemble_search_results = ensemble_search_results[: self.k]

        if self.reorder is not None:
            ensemble_search_results = self.reorder.transform_documents(
                ensemble_search_results
            )

        if self.verbose:
            log_or_print(
                f"Ensemble search results: {pformat(ensemble_search_results)}",
                logger=self.logger,
            )

        return [result[0] for result in ensemble_search_results]


def get_embeddings(
    model_id: str,
    endpoint_name: Optional[str],
    max_str_len: Optional[int] = None,
    profile_name: Optional[str] = None,
    region_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Embeddings:
    if get_provider_name(model_id) in ("amazon", "cohere"):
        embeddings = BedrockEmbeddingsWrapper(
            credentials_profile_name=profile_name,
            region_name=region_name,
            model_id=model_id,
            max_str_len=max_str_len,
        )

    elif "text-embedding" in model_id:
        embeddings = OpenAIEmbeddings(
            model=model_id, dimensions=get_dimensions(model_id)
        )

    else:
        if endpoint_name is None:
            embeddings = HuggingFaceEmbeddings(model_name=model_id)

        else:
            embeddings = SagemakerEndpointEmbeddingsWrapper(
                endpoint_name=endpoint_name,
                region_name=region_name,
                content_handler=EmbeddingsContentHandlerWrapper(),
                max_str_len=max_str_len,
                logger=logger,
            )

    return embeddings


def get_llm(
    model_id: str,
    temperature: float,
    profile_name: Optional[str] = None,
    region_name: Optional[str] = None,
    top_k: int = 50,
    top_p: float = 0.95,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    streaming: bool = False,
) -> BaseChatModel:
    if model_id in [
        ChatModelId.CLAUDE_V1,
        ChatModelId.CLAUDE_V2_1,
        ChatModelId.CLAUDE_V3_HAIKU,
        ChatModelId.CLAUDE_V3_SONNET,
        ChatModelId.CLAUDE_V3_OPUS,
    ]:
        return BedrockChat(
            callbacks=callbacks,
            credentials_profile_name=profile_name,
            region_name=region_name,
            model_id=model_id,
            streaming=streaming,
            model_kwargs={
                "max_tokens": 4096,
                "stop_sequences": ["\n\nHuman:"],
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            },
        )

    if model_id in [
        ChatModelId.GPT_3_5_TURBO,
        ChatModelId.GPT_4_TURBO,
    ]:
        return ChatOpenAI(
            callbacks=callbacks,
            max_tokens=1024,
            model_name=model_id,
            streaming=streaming,
            temperature=temperature,
        )

    raise ValueError(f"Invalid model_id: {model_id}")


def get_memory(
    memory_type,
    memory_key: Optional[str] = None,
    human_prefix: Optional[str] = None,
    ai_prefix: Optional[str] = None,
    return_messages: bool = True,
    k: int = 5,
    llm: Optional[BaseChatModel] = None,
):
    memory_type = get_default(memory_type, "ConversationBufferMemory")
    human_prefix = get_default(human_prefix, "Human")
    ai_prefix = get_default(ai_prefix, "AI")
    memory_key = get_default(memory_key, "chat_history")

    if memory_type == "ConversationBufferMemory":
        memory = ConversationBufferMemory(
            human_prefix=human_prefix,
            ai_prefix=ai_prefix,
            memory_key=memory_key,
            return_messages=return_messages,
            output_key="answer",
        )

    elif memory_type == "ConversationBufferWindowMemory":
        memory = ConversationBufferWindowMemory(
            human_prefix=human_prefix,
            ai_prefix=ai_prefix,
            memory_key=memory_key,
            return_messages=return_messages,
            k=k,
            output_key="answer",
        )

    elif memory_type == "ConversationSummaryBufferMemory":
        assert (
            llm is not None
        ), "LLM muse be provided for 'ConversationSummaryBufferMemory'."

        memory = ConversationSummaryBufferMemory(
            human_prefix=human_prefix,
            ai_prefix=ai_prefix,
            memory_key=memory_key,
            return_messages=return_messages,
            llm=llm,
            max_token_limit=1024,
            prompt=get_summary_prompt(),
            output_key="answer",
        )

    else:
        raise ValueError(f"Invalid memory type: {memory_type}")

    return memory


def get_retriever(
    profile_name: Optional[str],
    region_name: Optional[str],
    domain_name: str,
    index_name: str,
    endpoint_name: Optional[str],
    secret_name: str,
    embedding_model_id: str,
    retriever_type: str,
    k: int,
    multiplier: int,
    semantic_weight: float,
    use_hyde: bool,
    use_rag_fusion: bool,
    use_parent_document: bool,
    use_time_weight: bool,
    use_reorder: bool,
    decay_rate: float,
    temperature: float,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> BaseRetriever:
    boto_session = boto3.Session(region_name=region_name, profile_name=profile_name)
    domain_endpoint = get_opensearch_domain_endpoint(boto_session, domain_name)

    secret = get_secret(boto_session, secret_name)
    opensearch_client = create_opensearch_client(
        domain_endpoint,
        (secret["opensearch_username"], secret["opensearch_password"]),
    )

    boolean_filter = (
        [{"term": {"metadata.hierarchy": "child"}}] if use_parent_document else []
    )
    reorder = LongContextReorder() if use_reorder else None

    if retriever_type in ["lexical", "ensemble"]:
        lexical_retriever = OpenSearchRetriever(
            opensearch_client=opensearch_client,
            index_name=index_name,
            k=k,
            retriever_type=retriever_type,
            boolean_filter=boolean_filter,
            use_parent_document=use_parent_document,
            reorder=reorder,
            verbose=verbose,
            logger=logger,
        )

    else:
        lexical_retriever = None

    if retriever_type in ["semantic", "ensemble"]:
        embeddings = get_embeddings(
            embedding_model_id,
            endpoint_name,
            profile_name=profile_name,
            region_name=region_name,
            logger=logger,
        )

        semantic_retriever = OpenSearchRetriever(
            opensearch_client=opensearch_client,
            index_name=index_name,
            embeddings=embeddings,
            k=k,
            retriever_type=retriever_type,
            boolean_filter=boolean_filter,
            use_parent_document=use_parent_document,
            reorder=reorder,
            verbose=verbose,
            logger=logger,
        )

    else:
        embeddings = None
        semantic_retriever = None

    if retriever_type == "lexical":
        retriever = lexical_retriever

    elif retriever_type == "semantic":
        retriever = semantic_retriever

    elif retriever_type == "ensemble":
        model_id = ChatModelId.CLAUDE_V3_SONNET
        retriever = OpenSearchEnsembleRetriever(
            opensearch_client=opensearch_client,
            index_name=index_name,
            embeddings=embeddings,
            llm=get_llm(
                model_id,
                temperature,
                profile_name=profile_name,
                region_name=region_name,
            ),
            k=k,
            multiplier=multiplier,
            boolean_filter=boolean_filter,
            weights=[1.0 - semantic_weight, semantic_weight],
            is_async=True,
            use_hyde=use_hyde,
            use_rag_fusion=use_rag_fusion,
            use_parent_document=use_parent_document,
            use_time_weight=use_time_weight,
            decay_rate=decay_rate,
            reorder=reorder,
            profile_name=profile_name,
            region_name=region_name,
            verbose=verbose,
            logger=logger,
        )

    else:
        raise ValueError(f"Invalid retriever type: {retriever_type}")

    return retriever


def normalize_search_results(search_results: Dict[str, Any]) -> Dict[str, Any]:
    hits = search_results["hits"]["hits"]
    max_score = float(search_results["hits"]["max_score"])
    for hit in hits:
        hit["_score"] = float(hit["_score"]) / max_score
    search_results["hits"]["max_score"] = hits[0]["_score"]
    search_results["hits"]["hits"] = hits
    return search_results


def search_in_opensearch(
    opensearch_client: OpenSearch,
    index_name: str,
    query: str,
    retriever_type: Optional[str] = None,
    k: int = 5,
    boolean_filter: Optional[List[Dict[str, Dict[str, str]]]] = None,
    minimum_should_match: int = 0,
    embeddings: Optional[Embeddings] = None,
    use_ensemble: bool = False,
) -> Union[List[Document], List[Tuple[Document, float]]]:
    if retriever_type.lower() == "lexical":
        query = get_opensearch_query(
            query,
            retriever_type=retriever_type,
            k=k,
            boolean_filter=boolean_filter,
            minimum_should_match=minimum_should_match,
        )

    elif retriever_type.lower() == "semantic":
        assert (
            embeddings is not None
        ), "An embedding model must be provided for semantic search."

        query = get_opensearch_query(
            query=query,
            retriever_type=retriever_type,
            k=k,
            boolean_filter=boolean_filter,
            vector_field="vector_field",
            vector=embeddings.embed_query(query),
        )

    else:
        ValueError(f"Invalid retriever_type: {retriever_type}")

    query["size"] = k
    search_results = opensearch_client.search(body=query, index=index_name)

    results = []
    if (
        search_results["hits"]["hits"] is not None
        and len(search_results["hits"]["hits"]) > 0
    ):
        search_results = normalize_search_results(search_results)
        for result in search_results["hits"]["hits"]:
            metadata = result["_source"]["metadata"]
            metadata["id"] = result["_id"]

            doc = Document(page_content=result["_source"]["text"], metadata=metadata)
            if use_ensemble:
                results.append((doc, result["_score"]))
            else:
                results.append(doc)

    return results


def describe_docs_stat(
    docs: Iterable[Document], logger: Optional[logging.Logger] = None
) -> None:
    log_or_print(
        str(
            pd.DataFrame(
                [len(doc.page_content) for doc in docs], columns=["summary"]
            ).describe()
        ),
        logger=logger,
    )
    avg_n_char_per_doc = round(sum(len(doc.page_content) for doc in docs) / len(docs))
    log_or_print(
        f"Average number of characters per document is {avg_n_char_per_doc}.",
        logger=logger,
    )


def show_docs(
    docs: Iterable[Document], limit: int = 10, logger: Optional[logging.Logger] = None
):
    for idx, doc in enumerate(docs):
        if idx < limit:
            string = (
                "-" * 100
                + "\npage_content:\n"
                + doc.page_content
                + "\nmetadata: "
                + str(doc.metadata)
            )
            log_or_print(
                string,
                logger=logger,
            )
        else:
            break


def load_docs_from_jsonl(file_path: str) -> Iterable[Document]:
    docs = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            docs.append(Document(**json.loads(line)))
    return docs


def save_docs_to_jsonl(docs: Iterable[Document], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        for doc in docs:
            file.write(doc.json() + "\n")
