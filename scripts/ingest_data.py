import glob
import logging
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from json.decoder import JSONDecodeError
from typing import Any, Dict, Final, List, Optional, Tuple
import boto3
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    LLMSherpaFileLoader,
    PyMuPDFLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
)
from langchain_community.vectorstores import OpenSearchVectorSearch
from urllib3.exceptions import MaxRetryError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from app.utils import (
    create_opensearch_client,
    describe_docs_stat,
    download_dir_from_s3,
    get_date_values,
    get_dimensions,
    get_embeddings,
    get_opensearch_domain_endpoint,
    get_model_id,
    get_name,
    get_secret,
    get_ssm_param_key,
    get_ssm_param_value,
    load_config,
    log_or_print,
    logger,
    make_s3_uri,
    save_docs_to_jsonl,
    show_docs,
)


def clean_doc(docs: List[Document]) -> List[Document]:
    email_pattern = r"\b[A-Za-z0-9._%+-]+[ ]*@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    phone_pattern = r"\d{2,3}-\d{3,4}-\d{4}"
    name_patterns = [
        r"\b([가-힣]+\s+선임연구원)",
        r"\b(Analyst\s+[가-힣]+)",
        r"\b(RA\s+[가-힣]+)",
        r"\b(애널리스트\s+\([가-힣]+\s+\))",
    ]

    cleaned_docs = []
    for doc in docs:
        doc.page_content = re.sub(email_pattern, "", doc.page_content).strip()
        doc.page_content = re.sub(phone_pattern, "", doc.page_content).strip()
        for name_pattern in name_patterns:
            doc.page_content = re.sub(name_pattern, "", doc.page_content).strip()
        cleaned_docs.append(doc)

    return cleaned_docs


def get_dir_path(dir_name: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), dir_name))


def get_index_body(dimensions: int) -> Dict[str, Any]:
    index_body = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "my_analyzer": {
                        "char_filter": ["html_strip"],
                        "tokenizer": "nori",
                        "filter": [
                            "nori_number",
                            "lowercase",
                            "trim",
                            "my_nori_part_of_speech",
                        ],
                        "type": "custom",
                    }
                },
                "tokenizer": {
                    "nori": {
                        "decompound_mode": "mixed",
                        "discard_punctuation": "true",
                        "type": "nori_tokenizer",
                    }
                },
                "filter": {
                    "my_nori_part_of_speech": {
                        "type": "nori_part_of_speech",
                        "stoptags": [
                            "E",
                            "IC",
                            "J",
                            "MAG",
                            "MAJ",
                            "MM",
                            "NA",
                            "SC",
                            "SE",
                            "SP",
                            "SSC",
                            "SSO",
                            "XPN",
                            "XSA",
                            "XSN",
                            "XSV",
                            "UNA",
                            "VSV",
                        ],
                    }
                },
            },
            "index": {
                "knn": True,
                "knn.space_type": "cosinesimil",
            },
        },
        "mappings": {
            "properties": {
                "metadata": {
                    "properties": {
                        "last_updated": {"type": "date"},
                        "project": {"type": "keyword"},
                        "seq_num": {"type": "long"},
                        "source": {"type": "keyword"},
                        "title": {"type": "text"},
                        "url": {"type": "text"},
                    }
                },
                "text": {
                    "analyzer": "my_analyzer",
                    "search_analyzer": "my_analyzer",
                    "type": "text",
                },
                "vector_field": {
                    "type": "knn_vector",
                    "dimension": dimensions,
                },
            }
        },
    }
    return index_body


def parse_pdf(
    raw_data_dir: str,
    pdf_parser_type: str,
    merge_docs_from_pdf: bool = False,
    host_api: bool = False,
    logger: Optional[logging.Logger] = None,
) -> List[Document]:
    doc_paths = glob.glob(os.path.join(raw_data_dir, "*", "*.pdf"))
    docs: List[Document] = []
    parser_type = pdf_parser_type.lower()

    loader_map = {
        "layout_pdf_reader": LLMSherpaFileLoader,
        "pymupdf": PyMuPDFLoader,
        "pypdf": PyPDFLoader,
        "unstructured": UnstructuredPDFLoader,
    }

    if parser_type not in loader_map:
        raise ValueError(f"Invalid PDF parser type provided: '{pdf_parser_type}'")

    for doc_path in doc_paths:
        doc_filename = os.path.basename(doc_path)
        log_or_print(f"Parsing of '{doc_filename}' started.", logger=logger)

        try:
            kwargs = (
                {
                    "llmsherpa_api_url": "http://localhost:5001/api/parseDocument?renderFormat=all"
                    if host_api
                    else "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
                }
                if pdf_parser_type == "layout_pdf_reader"
                else {}
            )
            loader = loader_map[parser_type](doc_path, **kwargs)
            doc = loader.load()

            if len(doc) == 0:
                loader = PyPDFLoader(doc_path)
                doc = loader.load()
                log_or_print(
                    f"Parsing of '{doc_filename}' was reprocessed through 'pypdf'.",
                    logger=logger,
                )

        except (JSONDecodeError, KeyError, MaxRetryError):
            loader = PyPDFLoader(doc_path)
            doc = loader.load()
            log_or_print(
                f"Parsing of '{doc_filename}' was reprocessed through 'pypdf'.",
                logger=logger,
            )

        if merge_docs_from_pdf:
            page_content = "".join(chunk.page_content for chunk in doc)
            doc = [Document(page_content=page_content, metadata=doc[0].metadata)]

        docs.extend(doc)
        log_or_print(f"Parsing of '{doc_filename}' completed.", logger=logger)

    return docs


def _parse_pdf_wrapper(
    args: Tuple[str, str, bool, bool, Optional[logging.Logger]]
) -> List[Document]:
    return parse_pdf(*args)


def parse_pdf_in_parallel(
    raw_data_dirs: List[str],
    pdf_parser_type: str,
    merge_docs_from_pdf: bool = False,
    host_api: bool = False,
    max_workers: int = 4,
    logger: Optional[logging.Logger] = None,
) -> List[Document]:
    if max_workers > 1:
        tasks = [
            (raw_data_dir, pdf_parser_type, merge_docs_from_pdf, host_api, logger)
            for raw_data_dir in raw_data_dirs
        ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_parse_pdf_wrapper, task) for task in tasks]
            docs = []
            for future in futures:
                docs.extend(future.result())

    else:
        docs = []
        for raw_data_dir in raw_data_dirs:
            docs.extend(
                parse_pdf(
                    raw_data_dir, pdf_parser_type, merge_docs_from_pdf, logger=logger
                )
            )

    return docs


CATEGORIES: Final = ["company", "debenture", "economy", "industry", "invest", "market"]
FRAC_CHAR_TO_TOKEN: Final = 0.4
MAX_WORKERS: Final = 4


if __name__ == "__main__":
    # Set the start and end dates of the web scraping
    START_DATE, END_DATE = "2024-05-20", "2024-05-20"

    config_dir = get_dir_path(os.path.join(os.pardir, "app", "configs"))
    config = load_config(os.path.join(config_dir, "config.yaml"))

    boto_session = boto3.Session(
        region_name=config.region_name, profile_name=config.profile_name
    )
    bucket_name = get_ssm_param_value(
        boto_session,
        get_ssm_param_key(config.proj_name, config.stage, "s3_bucket", "default"),
    )

    raw_data_dir = os.path.join(os.pardir, "raw_data")
    date_values = get_date_values(START_DATE, END_DATE)

    if os.path.exists(raw_data_dir):
        shutil.rmtree(raw_data_dir)

    # Download PDF documents from the S3 bucket
    for category in CATEGORIES:
        for date_value in date_values:
            prefix = f"raw_data/{category}/{date_value}"
            download_dir_from_s3(
                boto_session,
                bucket_name,
                prefix,
                os.path.join(raw_data_dir, category, date_value),
                logger=logger,
            )
            logger.info(
                "Downloaded all documents from '%s'.",
                make_s3_uri(bucket_name, prefix),
            )

    if len(glob.glob(os.path.join(raw_data_dir, "*", "*", "*.pdf"))) > 0:
        # Set up the relevant resources
        ddb_resource = boto_session.resource("dynamodb")
        metadata_table = ddb_resource.Table(
            get_ssm_param_value(
                boto_session,
                get_ssm_param_key(
                    config.proj_name, config.stage, "ddb_table", "metadata"
                ),
            )
        )

        model_id = get_model_id(config.embedding_model_name)
        endpoint_name = get_name(config.proj_name, config.stage, "embedding")
        dimensions = get_dimensions(model_id)

        embeddings = get_embeddings(
            model_id,
            endpoint_name,
            max_str_len=int(FRAC_CHAR_TO_TOKEN * dimensions),
            profile_name=config.profile_name,
            region_name=config.region_name,
            logger=logger,
        )
        logger.info("The embedding model '%s' has been set for retrieval.", model_id)

        domain_name = get_name(config.proj_name, config.stage, "docs")
        domain_endpoint = get_opensearch_domain_endpoint(boto_session, domain_name)
        logger.info(
            "You can access the OpenSearch dashboard at the following link: %s",
            domain_endpoint + "/_dashboards/",
        )

        secret = get_secret(
            boto_session, get_name(config.proj_name, config.stage, "docs")
        )
        opensearch_client = create_opensearch_client(
            domain_endpoint,
            (secret["opensearch_username"], secret["opensearch_password"]),
        )

        # Created an OpenSearch index or deleted documents for the dates
        index_name = get_name(config.proj_name, config.stage, "docs")
        if not opensearch_client.indices.exists(index=index_name):
            opensearch_client.indices.create(
                index_name,
                body=get_index_body(dimensions),
            )

            logger.info(
                "The OpenSearch index '%s' has been created. (dimension size: %d)",
                index_name,
                dimensions,
            )

        else:
            n = 0
            for date_value in date_values:
                response = opensearch_client.delete_by_query(
                    index=index_name,
                    body={"query": {"match": {"metadata.base_date": date_value}}},
                )
                n += response["deleted"]

            logger.info(
                "%d documents were deleted from the OpenSearch index '%s'.",
                n,
                index_name,
            )

        # Delete the existing local directory and create a new one
        proc_data_dir = get_dir_path(os.path.join(os.pardir, "proc_data"))
        parent_docs_path = os.path.join(proc_data_dir, "parent_docs.jsonl")
        child_docs_path = os.path.join(proc_data_dir, "child_docs.jsonl")

        if os.path.exists(proc_data_dir):
            shutil.rmtree(proc_data_dir)
        os.makedirs(proc_data_dir)

        # Parse PDF files as documents
        raw_data_dirs = [
            get_dir_path(os.path.join(os.pardir, "raw_data", category))
            for category in CATEGORIES
        ]

        parent_docs = parse_pdf_in_parallel(
            raw_data_dirs,
            config.pdf_parser_type,
            merge_docs_from_pdf=config.use_parent_document,
            host_api=True,
            max_workers=MAX_WORKERS,
            logger=logger,
        )

        logger.info(
            "The documents were successfully parsed using '%s'.", config.pdf_parser_type
        )

        # Clean documents
        parent_docs = clean_doc(parent_docs)

        # Insert metadata into documents
        for parent_doc in parent_docs:
            source = parent_doc.metadata["source"]
            doc_id = source.split("/")[-1].split(".")[0]
            response = metadata_table.get_item(Key={"DocId": doc_id})
            item = response.get("Item")

            parent_doc.metadata["source"] = make_s3_uri(
                bucket_name,
                config.proj_name,
                filename="/".join(source.split("/")[-3:]),
            )
            parent_doc.metadata["hierarchy"] = "parent"
            parent_doc.metadata["parent"] = None

            for field, attr in zip(
                [
                    "category",
                    "title",
                    "publisher",
                    "base_date",
                ],
                [
                    "Category",
                    "Title",
                    "Publisher",
                    "BaseDate",
                ],
            ):
                parent_doc.metadata[field] = item[attr]

            if item.get("Item") is not None:
                parent_doc.metadata["item"] = item["Item"]

        # Ingest parent documents into the OpenSearch index
        vector_store = OpenSearchVectorSearch(
            domain_endpoint,
            index_name,
            embeddings,
            http_auth=(secret["opensearch_username"], secret["opensearch_password"]),
            is_aoss=False,
            engine="faiss",
            space_type="l2",
            bulk_size=100000,
            timeout=60,
        )

        if config.use_parent_document:
            if not (
                config.parent_chunk_size is None or config.parent_chunk_overlap is None
            ):
                # Split parent documents
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ".", " ", ""],
                    chunk_size=config.parent_chunk_size,
                    chunk_overlap=config.parent_chunk_overlap,
                    length_function=len,
                )
                parent_docs = text_splitter.split_documents(parent_docs)

            doc_ids = vector_store.add_documents(
                parent_docs, vector_field="vector_field", bulk_size=100000
            )

            logger.info(
                "%d parent documents have been ingested into the OpenSearch index '%s'.",
                len(doc_ids),
                index_name,
            )

        else:
            doc_ids = [None] * len(parent_docs)

        describe_docs_stat(parent_docs, logger=logger)
        show_docs(parent_docs, limit=1, logger=logger)
        save_docs_to_jsonl(parent_docs, parent_docs_path)
        logger.info(
            "The parent documents were saved to '%s'.",
            parent_docs_path,
        )

        # Split child documents
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " ", ""],
            chunk_size=config.child_chunk_size,
            chunk_overlap=config.child_chunk_overlap,
            length_function=len,
        )

        child_docs = []
        for doc_id, parent_doc in zip(doc_ids, parent_docs):
            for child_doc in text_splitter.split_documents([parent_doc]):
                child_doc.metadata["hierarchy"] = "child"
                child_doc.metadata["parent"] = doc_id
                child_docs.append(child_doc)

        # Ingest child documents into the OpenSearch index
        doc_ids = vector_store.add_documents(
            child_docs, vector_field="vector_field", bulk_size=100000
        )

        describe_docs_stat(child_docs, logger=logger)
        show_docs(child_docs, limit=5, logger=logger)

        save_docs_to_jsonl(child_docs, child_docs_path)
        logger.info("The child documents were saved to '%s'.", child_docs_path)

        logger.info(
            "%d child documents have been ingested into the OpenSearch index '%s'.",
            len(doc_ids),
            index_name,
        )
