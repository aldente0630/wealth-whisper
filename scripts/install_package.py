import os
import sys
import boto3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from app.utils import (
    create_opensearch_client,
    get_name,
    get_opensearch_domain_endpoint,
    get_secret,
    load_config,
    logger,
    wait_for_opensearch_package_association,
)


def get_dir_path(dir_name: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), dir_name))


def get_nori_package_id(region_name: str, opensearch_version: str) -> str:
    nori_package_id_dict = {
        "us-east-1": {
            "2.3": "G196105221",
            "2.5": "G240285063",
            "2.7": "G16029449",
            "2.9": "G60209291",
            "2.11": "G181660338",
        },
        "us-west-2": {
            "2.3": "G94047474",
            "2.5": "G138227316",
            "2.7": "G182407158",
            "2.9": "G226587000",
            "2.11": "G79602591",
        },
    }

    return nori_package_id_dict[region_name][opensearch_version]


if __name__ == "__main__":
    config_dir = get_dir_path(os.path.join(os.pardir, "app", "configs"))
    config = load_config(os.path.join(config_dir, "config.yaml"))

    boto_session = boto3.Session(
        region_name=config.region_name, profile_name=config.profile_name
    )

    # Install the Nori plugin
    opensearch_client = boto_session.client("opensearch")
    domain_name = get_name(config.proj_name, config.stage, "docs")
    opensearch_domain_endpoint = get_opensearch_domain_endpoint(
        boto_session, domain_name
    )

    _ = opensearch_client.associate_package(
        PackageID=get_nori_package_id(config.region_name, "2.11"),
        DomainName=domain_name,
    )
    wait_for_opensearch_package_association(boto_session, domain_name, logger=logger)

    # Verify that it installed correctly
    secret = get_secret(boto_session, get_name(config.proj_name, config.stage, "docs"))
    opensearch_client = create_opensearch_client(
        opensearch_domain_endpoint,
        (secret["opensearch_username"], secret["opensearch_password"]),
    )
    result = opensearch_client.cat.plugins()

    logger.info(
        "The OpenSearch Nori plugin is available."
        if "opensearch-analysis-nori" in result
        else "The OpenSearch Nori plugin failed to install."
    )
