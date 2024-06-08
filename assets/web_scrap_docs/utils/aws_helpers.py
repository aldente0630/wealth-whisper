import logging
import os
import sys
from typing import Dict, List, Optional
import boto3
from botocore.exceptions import ClientError

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from .misc import log_or_print


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
