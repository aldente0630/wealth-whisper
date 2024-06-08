import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import boto3
import numpy as np
from botocore.exceptions import ClientError
from langchain_community.embeddings.sagemaker_endpoint import (
    EmbeddingsContentHandler,
    SagemakerEndpointEmbeddings,
)
from opensearchpy import OpenSearch, RequestsHttpConnection

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from app.utils.misc import get_default, log_or_print


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


def delete_files_in_s3(
    boto_session: boto3.Session,
    bucket_name: str,
    prefix: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    s3_resource = boto_session.resource("s3")
    bucket = s3_resource.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=prefix):
        s3_resource.Object(bucket_name, obj.key).delete()
        log_or_print(
            f"The 's3://{bucket_name}/{obj.key}' file has been deleted.", logger
        )


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


def get_account_id(boto_session: boto3.Session) -> str:
    sts_client = boto_session.client("sts")
    try:
        response = sts_client.get_caller_identity()
        return response["Account"]

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


def get_opensearch_query(
    query: str,
    retriever_type: Optional[str] = None,
    k: Optional[int] = 5,
    boolean_filter: Optional[List[Dict[str, Dict[str, str]]]] = None,
    minimum_should_match: int = 0,
    vector_field: Optional[str] = None,
    vector: Optional[List[float]] = None,
) -> Dict[str, Any]:
    retriever_type = get_default(retriever_type, "lexical")
    vector_field = get_default(vector_field, "vector_field")

    if retriever_type.lower() == "lexical":
        query_template = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "text": {
                                    "query": query,
                                    "minimum_should_match": f"{minimum_should_match}%",
                                    "operator": "or",
                                    # "boost": 1
                                    # "fuzziness": "AUTO",
                                    # "fuzzy_transpositions": True,
                                    # "lenient": False,
                                    # "max_expansions": 50,
                                    # "prefix_length": 0,
                                    # "zero_terms_query": "none",
                                }
                            }
                        },
                    ],
                    "filter": get_default(boolean_filter, []),
                }
            }
        }

    elif retriever_type.lower() == "semantic":
        query_template = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                vector_field: {
                                    "vector": vector,
                                    "k": k,
                                }
                            }
                        },
                    ],
                    "filter": get_default(boolean_filter, []),
                }
            }
        }

    else:
        query_template = None
        ValueError(f"Invalid retriever_type: {retriever_type}")

    return query_template


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


def invoke_sagemaker_endpoint(
    boto_session: boto3.Session,
    endpoint_name: str,
    payload: Any,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    sm_runtime_client = boto_session.client("sagemaker-runtime")

    try:
        response = sm_runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps({"inputs": payload}),
        )

        result = json.loads(response["Body"].read().decode())
        return result
    except ClientError:
        log_or_print("The requested endpoint name doesn't exist.", logger=logger)
        return None


def make_s3_uri(bucket_name: str, prefix: str, filename: Optional[str] = None) -> str:
    prefix = prefix if filename is None else os.path.join(prefix, filename)
    return f"s3://{bucket_name}/{prefix}"


def submit_batch_job(
    boto_session: boto3.Session,
    job_name: str,
    job_queue_name: str,
    job_definition_name: str,
    parameters: Optional[Dict[str, str]] = None,
    logger: Optional[logging.Logger] = None,
):
    batch_client = boto_session.client("batch")
    try:
        _ = batch_client.submit_job(
            jobName=job_name,
            jobQueue=job_queue_name,
            jobDefinition=job_definition_name,
            parameters=parameters,
        )
        log_or_print(f"The batch job '{job_name}' was submitted successfully.", logger)

    except ClientError as error:
        raise error


def upload_dir_to_s3(
    boto_session: boto3.Session,
    local_dir: str,
    bucket_name: str,
    prefix: str,
    file_ext_to_excl: Optional[List[str]] = None,
    public_readable: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    s3_client = boto_session.client("s3")
    file_ext_to_excl = [] if file_ext_to_excl is None else file_ext_to_excl
    extra_args = {"ACL": "public-read"} if public_readable else {}

    for root, _, files in os.walk(local_dir):
        for file in files:
            if file.split(".")[-1] not in file_ext_to_excl:
                try:
                    s3_client.upload_file(
                        os.path.join(root, file),
                        bucket_name,
                        f"{prefix}/{file}",
                        ExtraArgs=extra_args,
                    )
                    log_or_print(
                        f"The '{file}' file has been uploaded to 's3://{bucket_name}/{prefix}/{file}'.",
                        logger,
                    )

                except ClientError as error:
                    raise error


def upload_file_to_s3(
    boto_session: boto3.Session,
    local_file_path: str,
    bucket_name: str,
    prefix: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    s3_client = boto_session.client("s3")

    try:
        filename = os.path.basename(local_file_path)
        s3_client.upload_file(
            local_file_path,
            bucket_name,
            f"{prefix}/{filename}",
        )
        log_or_print(
            f"The '{filename}' file has been uploaded to 's3://{bucket_name}/{prefix}'.",
            logger,
        )

    except ClientError as error:
        raise error


def wait_for_opensearch_package_association(
    boto_session: boto3.Session,
    domain_name: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    opensearch_client = boto_session.client("opensearch")
    try:
        response = opensearch_client.list_packages_for_domain(
            DomainName=domain_name, MaxResults=1
        )
        while (
            response["DomainPackageDetailsList"][0]["DomainPackageStatus"]
            == "ASSOCIATING"
        ):
            log_or_print(
                "The package is being associated with the Opensearch domain...",
                logger=logger,
            )
            time.sleep(60)
            response = opensearch_client.list_packages_for_domain(
                DomainName=domain_name, MaxResults=1
            )

        log_or_print(
            "The package is associated with the Opensearch domain.", logger=logger
        )

    except ClientError as error:
        raise error
