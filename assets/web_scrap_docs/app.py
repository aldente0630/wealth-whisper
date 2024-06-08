import json
import os
import sys
from datetime import datetime
from typing import Any, List, Dict, Final, Union
import boto3
import requests
from bs4 import BeautifulSoup
from pytz import timezone

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils import (
    delete_files_in_s3,
    download_file_from_s3,
    get_date_values,
    get_name,
    get_ssm_param_key,
    get_ssm_param_value,
    load_config,
    logger,
    make_s3_uri,
    shift_datetime,
    submit_batch_job,
    upload_dir_to_s3,
)


def get_page_url(category: str, start_date: str, end_date: str, page_num: int) -> str:
    base_url = "finance.naver.com/research"
    category_dict = {
        "company": "company_list",
        "debenture": "debenture_list",
        "economy": "economy_list",
        "industry": "industry_list",
        "invest": "invest_list",
        "market": "market_info_list",
    }

    return f"https://{base_url}/{category_dict[category]}.naver?searchType=writeDate&writeFromDate={start_date}&writeToDate={end_date}&page={page_num}"


def parse_page(curr_page: List, category: str) -> List[Dict[str, Union[str, int]]]:
    props = []
    for row in curr_page:
        try:
            prop = (
                {
                    "item": row.find_all("td")[0].get_text(),
                    "title": row.find_all("td")[1].text.strip(),
                    "publisher": row.find_all("td")[2].text.strip(),
                }
                if category in ["company", "industry"]
                else {
                    "title": row.find("a").get_text(),
                    "publisher": row.find_all("td")[1].text.strip(),
                }
            )

            prop["pdf_url"] = row.find("td", class_="file").find("a", href=True)["href"]
            prop["date"] = row.find("td", class_="date").get_text()
            prop["view_count"] = row.find_all("td", class_="date")[1].get_text()
            props.append(prop)

        except (AttributeError, IndexError, TypeError):
            continue

    return props


CATEGORIES: Final = ["company", "debenture", "economy", "industry", "invest", "market"]
TEMP_DIR: Final = "/tmp"


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    # Load the configuration file
    proj_name = os.environ["PROJ_NAME"]
    stage = os.getenv("STAGE", "dev")
    region_name = os.getenv("REGION_NAME", "us-east-1")

    boto_session = boto3.Session(region_name=region_name)
    bucket_name = get_ssm_param_value(
        boto_session, get_ssm_param_key(proj_name, stage, "s3_bucket", "default")
    )

    config_dir = os.path.join(TEMP_DIR, "configs")
    download_file_from_s3(
        boto_session,
        bucket_name,
        "configs",
        "config.yaml",
        config_dir,
        logger=logger,
    )
    config = load_config(os.path.join(config_dir, "config.yaml"))

    # Set the start and end dates for the web scraping
    end_date = event.get(
        "END_DATE",
        shift_datetime(
            datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d"),
            -int(config.idx_offset),
            freq=config.idx_freq,
            fmt="%Y-%m-%d",
        ),
    )
    start_date = event.get(
        "START_DATE",
        shift_datetime(
            end_date,
            -int(config.idx_duration),
            freq=config.idx_freq,
            fmt="%Y-%m-%d",
        ),
    )
    logger.info(
        "The web scraping start date is '%s' and the end date is '%s'.",
        start_date,
        end_date,
    )

    # Fetch the necessary information from web pages
    all_props = {}
    for category in CATEGORIES:
        props, prev_page, i = [], None, 0

        while True:
            page_url = get_page_url(category, start_date, end_date, i + 1)
            try:
                response = requests.get(page_url, timeout=10)
                if response.status_code != 200:
                    logger.error("There was an error fetching page '%s'.", page_url)
                    break

                soup = BeautifulSoup(response.text, "html.parser")
                curr_page = soup.find_all("tr")
                if curr_page == prev_page:
                    break

                props.extend(parse_page(curr_page, category))
                prev_page = curr_page
                i += 1

            except requests.RequestException as error:
                logger.error(
                    "The request for '%s' failed with exception '%s'.", page_url, error
                )
                break

        all_props[category] = props

    # Delete the existing metadata in the DynamoDB table
    ddb_resource = boto_session.resource("dynamodb")
    metadata_table = ddb_resource.Table(
        get_ssm_param_value(
            boto_session, get_ssm_param_key(proj_name, stage, "ddb_table", "metadata")
        )
    )
    index_name = get_ssm_param_value(
        boto_session, get_ssm_param_key(proj_name, stage, "ddb_index", "metadata")
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
                ":start_date": start_date.replace("-", ""),
                ":end_date": end_date.replace("-", ""),
            },
        )

        for item in response["Items"]:
            metadata_table.delete_item(Key={"DocId": item["DocId"]})
            n += 1

    logger.info(
        "%d items were deleted from the DynamoDB table '%s'.", n, metadata_table.name
    )

    # Download PDF documents and update the metadata in DynamoDB
    raw_data_dir = os.path.join(TEMP_DIR, "raw_data")
    creation_date = datetime.now().strftime("%Y%m%d")
    n = 0

    for category, props in all_props.items():
        for prop in props:
            pdf_url = prop["pdf_url"]

            try:
                docs_dir = os.path.join(
                    raw_data_dir, category, "20" + prop["date"].replace(".", "")
                )
                os.makedirs(docs_dir, exist_ok=True)

                response = requests.get(pdf_url, timeout=10)
                response.raise_for_status()

                filename = pdf_url.split("/")[-1]
                with open(os.path.join(docs_dir, filename), "wb") as file:
                    file.write(response.content)

                logger.info("Downloaded '%s' to '%s'.", filename, docs_dir)

                item = {
                    "DocId": filename.split(".")[0],
                    "Category": category,
                    "Title": prop["title"],
                    "Publisher": prop["publisher"],
                    "BaseDate": "20" + prop["date"].replace(".", ""),
                    "ViewCount": prop["view_count"],
                    "CreationDate": creation_date,
                }
                if prop.get("item") is not None:
                    item["Item"] = prop["item"].replace("\n", "")

                metadata_table.put_item(Item=item)
                n += 1

            except requests.RequestException as error:
                logger.error("The download request failed with exception '%s'.", error)

    logger.info("Downloaded %d documents.", n)

    # Upload PDF documents to the S3 bucket
    date_values = get_date_values(start_date, end_date)

    for category in CATEGORIES:
        for date_value in date_values:
            prefix = f"raw_data/{category}/{date_value}"
            delete_files_in_s3(boto_session, bucket_name, prefix, logger=logger)
            logger.info(
                "Deleted all documents in '%s'.", make_s3_uri(bucket_name, prefix)
            )

            upload_dir_to_s3(
                boto_session,
                os.path.join(raw_data_dir, category, date_value),
                bucket_name,
                prefix,
                logger=logger,
            )
            logger.info(
                "Uploaded all documents to '%s'.", make_s3_uri(bucket_name, prefix)
            )

    job_queue_name = get_ssm_param_value(
        boto_session,
        get_ssm_param_key(proj_name, stage, "batch_job_queue", "ingestion"),
    )
    job_definition_name = get_ssm_param_value(
        boto_session,
        get_ssm_param_key(proj_name, stage, "batch_job_definition", "ingestion"),
    )

    # Once the web scraping is done, start the data ingestion Batch job
    job_name = (
        get_name(proj_name, stage, "ingestion")
        + "-"
        + datetime.now(timezone("Asia/Seoul")).strftime("%Y%m%d%H%M%S")
    )
    submit_batch_job(
        boto_session,
        job_name,
        job_queue_name,
        job_definition_name,
        parameters={"start_date": start_date, "end_date": end_date},
        logger=logger,
    )

    return {
        "statusCode": 200,
        "body": json.dumps("The web scraping task has completed successfully."),
    }
