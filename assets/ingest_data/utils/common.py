from typing import Optional


def get_dimensions(model_id: str) -> int:
    dimensions_dict = {
        "amazon.titan-embed-g1-text-02": 1536,
        "BM-K/KoSimCSE-roberta": 768,
        "cohere.embed-multilingual-v3": 1024,
        "intfloat/multilingual-e5-large": 1024,
        "text-embedding-3-large": 3072,
    }
    return dimensions_dict[model_id]


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


def get_provider_name(model_id: str) -> str:
    return model_id.split(".")[0].split("/")[0]


def get_ssm_param_key(
    proj_name: str, stage: str, resource_name: str, resource_value: str
) -> str:
    return f"/{proj_name}/{stage}/{resource_name}/{resource_value}"
