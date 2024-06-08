import os
import sys
from typing import Final
import boto3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from app.utils import (
    create_opensearch_client,
    get_date_values,
    get_name,
    get_opensearch_domain_endpoint,
    get_secret,
    get_ssm_param_key,
    get_ssm_param_value,
    load_config,
    logger,
)


def get_dir_path(dir_name: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), dir_name))


CATEGORIES: Final = ["company", "debenture", "economy", "industry", "invest", "market"]

if __name__ == "__main__":
    # Set start and end dates for deleting documents
    START_DATE, END_DATE = "2024-05-15", "2024-05-19"

    config_dir = get_dir_path(os.path.join(os.pardir, "app", "configs"))
    config = load_config(os.path.join(config_dir, "config.yaml"))

    boto_session = boto3.Session(
        region_name=config.region_name, profile_name=config.profile_name
    )

    ddb_resource = boto_session.resource("dynamodb")
    metadata_table = ddb_resource.Table(
        get_ssm_param_value(
            boto_session,
            get_ssm_param_key(config.proj_name, config.stage, "ddb_table", "metadata"),
        )
    )
    index_name = get_ssm_param_value(
        boto_session,
        get_ssm_param_key(config.proj_name, config.stage, "ddb_index", "metadata"),
    )

    n = 0
    for category in CATEGORIES:
        response = metadata_table.query(
            IndexName=index_name,
            KeyConditionExpression="#category = :category AND #base_date BETWEEN :start_date AND :end_date",
            ExpressionAttributeNames={
                "#category": "Category",
                "#base_date": "BaseDate",
            },
            ExpressionAttributeValues={
                ":category": category,
                ":start_date": START_DATE.replace("-", ""),
                ":end_date": END_DATE.replace("-", ""),
            },
        )

        for item in response["Items"]:
            metadata_table.delete_item(Key={"DocId": item["DocId"]})
            n += 1

    logger.info(
        "%d items were deleted from the DynamoDB table '%s'.", n, metadata_table.name
    )

    domain_name = get_name(config.proj_name, config.stage, "docs")
    domain_endpoint = get_opensearch_domain_endpoint(boto_session, domain_name)
    secret = get_secret(boto_session, get_name(config.proj_name, config.stage, "docs"))

    opensearch_client = create_opensearch_client(
        domain_endpoint,
        (secret["opensearch_username"], secret["opensearch_password"]),
    )

    date_values = get_date_values(START_DATE, END_DATE)
    index_name = get_name(config.proj_name, config.stage, "docs")

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
