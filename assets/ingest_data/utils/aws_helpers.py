import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import boto3
import numpy as np
from botocore.exceptions import ClientError
from langchain_community.embeddings.sagemaker_endpoint import (
    EmbeddingsContentHandler,
    SagemakerEndpointEmbeddings,
)
from opensearchpy import OpenSearch, RequestsHttpConnection

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from .misc import log_or_print


class EmbeddingsContentHandlerWrapper(EmbeddingsContentHandler):
    content_type: str = "application/json"
    accepts: str = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict[str, Any]) -> bytes:
        input_json = json.dumps({"inputs": prompt, **model_kwargs})

        return input_json.encode("utf-8")

    def transform_output(self, output: bytes) -> Optional[List[List[float]]]:
        output_json = json.loads(output.read().decode("utf-8"))

        return [arr[0][0] for arr in np.array(output_json)]


class SagemakerEndpointEmbeddingsWrapper(SagemakerEndpointEmbeddings):
    exc_token: str = "<ERROR>"
    max_str_len: Optional[int] = None
    logger: Optional[logging.Logger] = None

    def embed_documents(
        self, texts: List[str], chunk_size: int = 1
    ) -> List[List[float]]:
        # Adjust chunk_size to not exceed the number of texts
        chunk_size = min(chunk_size, len(texts))

        results = []
        for i in range(0, len(texts), chunk_size):
            chunked_texts = texts[i : i + chunk_size]

            try:
                # Truncate texts if max_str_len is set
                if self.max_str_len is not None:
                    chunked_texts = [text[: self.max_str_len] for text in chunked_texts]

                response = self._embedding_func(chunked_texts)

            except ValueError as error:
                # Log the error and use the exception token for embedding
                log_or_print(
                    f"Input document: {chunked_texts}\nError occurred: {error}",
                    logger=self.logger,
                )
                response = self._embedding_func([self.exc_token])

            results.extend(response)

        return results


def create_opensearch_client(
    opensearch_domain_endpoint: str, http_auth: Tuple[str, str]
) -> OpenSearch:
    opensearch_client = OpenSearch(
        hosts=[
            {"host": opensearch_domain_endpoint.replace("https://", ""), "port": 443}
        ],
        http_auth=http_auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )

    return opensearch_client


def download_dir_from_s3(
    boto_session: boto3.Session,
    bucket_name: str,
    prefix: str,
    local_dir: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    s3_resource = boto_session.resource("s3")
    bucket = s3_resource.Bucket(bucket_name)
    os.makedirs(local_dir, exist_ok=True)

    for obj in bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith("/"):
            continue

        target_path = os.path.join(local_dir, os.path.relpath(obj.key, prefix))
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        bucket.download_file(obj.key, target_path)
        log_or_print(
            f"The '{obj.key}' file has been downloaded to '{target_path}'.", logger
        )


def download_file_from_s3(
    boto_session: boto3.Session,
    bucket_name: str,
    prefix: str,
    filename: str,
    local_dir: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    s3_client = boto_session.client("s3")
    try:
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, filename)
        s3_client.download_file(bucket_name, f"{prefix}/{filename}", local_path)
        log_or_print(
            f"The '{filename}' file has been downloaded to '{local_path}'.", logger
        )

    except ClientError as error:
        raise error


def get_opensearch_domain_endpoint(
    boto_session: boto3.Session, domain_name: str
) -> Optional[str]:
    opensearch_client = boto_session.client("opensearch")
    try:
        response = opensearch_client.describe_domain(DomainName=domain_name)
        return f"https://{response['DomainStatus']['Endpoint']}"

    except ClientError as error:
        raise error


def get_secret(boto_session: boto3.Session, secret_name: str) -> Dict[str, str]:
    secrets_client = boto_session.client(service_name="secretsmanager")
    try:
        response = secrets_client.get_secret_value(SecretId=secret_name)
        return json.loads(response["SecretString"])

    except ClientError as error:
        raise error


def get_ssm_param_value(boto_session: boto3.Session, param_name: str) -> str:
    ssm_client = boto_session.client("ssm")
    try:
        response = ssm_client.get_parameter(Name=param_name, WithDecryption=True)
        return response["Parameter"]["Value"]

    except ClientError as error:
        raise error


def make_s3_uri(bucket_name: str, prefix: str, filename: Optional[str] = None) -> str:
    prefix = prefix if filename is None else os.path.join(prefix, filename)
    return f"s3://{bucket_name}/{prefix}"
