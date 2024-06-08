import os
import sys
import boto3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from app.utils import (
    load_config,
    get_ssm_param_key,
    get_ssm_param_value,
    logger,
    upload_file_to_s3,
)


def get_dir_path(dir_name: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), dir_name))


if __name__ == "__main__":
    config_dir = get_dir_path(os.path.join(os.pardir, "app", "configs"))
    config = load_config(os.path.join(config_dir, "config.yaml"))
    config_path = get_dir_path(os.path.join(config_dir, "config.yaml"))
    boto_session = boto3.Session(
        region_name=config.region_name, profile_name=config.profile_name
    )
    bucket_name = get_ssm_param_value(
        boto_session,
        get_ssm_param_key(config.proj_name, config.stage, "s3_bucket", "default"),
    )

    upload_file_to_s3(
        boto_session,
        config_path,
        bucket_name,
        "configs",
        logger=logger,
    )
