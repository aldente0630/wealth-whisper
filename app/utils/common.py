from enum import Enum
from typing import Optional


class ChatModelId(str, Enum):
    CLAUDE_V1: str = "anthropic.claude-instant-v1"
    CLAUDE_V2_1: str = "anthropic.claude-v2:1"
    CLAUDE_V3_HAIKU: str = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_V3_SONNET: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_V3_OPUS: str = "anthropic.claude-3-opus-20240229-v1:0"
    GPT_3_5_TURBO: str = "gpt-3.5-turbo"
    GPT_4_TURBO: str = "gpt-4-turbo-preview"


def get_dimensions(model_id: str) -> int:
    dimensions_dict = {
        "amazon.titan-embed-g1-text-02": 1536,
        "BM-K/KoSimCSE-roberta": 768,
        "cohere.embed-multilingual-v3": 1024,
        "intfloat/multilingual-e5-large": 1024,
        "text-embedding-3-large": 3072,
    }
    return dimensions_dict[model_id]


def get_max_length(model_id: str) -> Optional[int]:
    max_length_dict = {
        "amazon.titan-embed-g1-text-02": 8192,
        "BM-K/KoSimCSE-roberta": 512,
        "cohere.embed-multilingual-v3": 1024,
        "intfloat/multilingual-e5-large": 512,
        "text-embedding-3-large": 8191,
    }
    return max_length_dict.get(model_id)


def get_model_id(model_name: str) -> Optional[str]:
    model_id_dict = {
        "amazon": "amazon.titan-embed-g1-text-02",
        "cohere": "cohere.embed-multilingual-v3",
        "e5": "intfloat/multilingual-e5-large",
        "openai": "text-embedding-3-large",
        "simcse": "BM-K/KoSimCSE-roberta",
    }
    return model_id_dict[model_name.lower()]


def get_name(proj_name: str, stage: str, resource_value: str) -> str:
    return "-".join([proj_name, stage, resource_value])


def get_ssm_param_key(
    proj_name: str, stage: str, resource_name: str, resource_value: str
) -> str:
    return f"/{proj_name}/{stage}/{resource_name}/{resource_value}"


def get_provider_name(model_id: str) -> str:
    return model_id.split(".")[0].split("/")[0]
