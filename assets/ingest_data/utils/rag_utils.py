import logging
import os
import sys
from typing import Iterable, List, Optional
import pandas as pd
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_community.embeddings import (
    BedrockEmbeddings,
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from .aws_helpers import (
    EmbeddingsContentHandlerWrapper,
    SagemakerEndpointEmbeddingsWrapper,
)
from .common import (
    get_dimensions,
    get_provider_name,
)
from .misc import log_or_print


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


def save_docs_to_jsonl(docs: Iterable[Document], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        for doc in docs:
            file.write(doc.json() + "\n")
