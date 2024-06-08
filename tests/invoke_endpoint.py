import os
import sys
import boto3
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from app.utils import (
    calc_cos_sim,
    get_model_id,
    get_name,
    invoke_sagemaker_endpoint,
    load_config,
    logger,
)


def get_dir_path(dir_name: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), dir_name))


if __name__ == "__main__":
    config_dir = get_dir_path(os.path.join(os.pardir, "app", "configs"))
    config = load_config(os.path.join(config_dir, "config.yaml"))

    boto_session = boto3.Session(
        region_name=config.region_name, profile_name=config.profile_name
    )

    # Test the SageMaker endpoint of the embedding model
    if get_model_id(config.embedding_model_name).split(".")[0].split("/")[
        0
    ] not in ("amazon", "cohere"):
        # Add your test samples
        payloads = [
            "미국의 물가상승률이 기대만큼 쉽게 내려오지 않고 있습니다.",
            "금일 KOSPI, KOSDAQ은 각각 -1.0%, -2.0% 하락했습니다.",
        ]

        endpoint_name = get_name(config.proj_name, config.stage, "embedding")
        responses = []
        for payload in payloads:
            logger.info("The original sentence: %s", payload)
            response = np.array(
                invoke_sagemaker_endpoint(
                    boto_session,
                    endpoint_name,
                    payload,
                    logger=logger,
                )
            )
            responses.append(response[0][0])

            logger.info("Dimension of the embedding vector: %s", response.shape)
            logger.info("The embedding vector value:\n%s", response)

        logger.info(
            "Cosine similarity between the first and the second sentences: %s",
            round(calc_cos_sim(responses[0], responses[1]), 6),
        )
