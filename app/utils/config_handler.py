import os
import sys
from dataclasses import dataclass
from typing import Optional
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from app.utils.misc import get_default


@dataclass
class Config:
    proj_name: str
    stage: str
    profile_name: Optional[str]
    region_name: str
    idx_duration: int
    idx_freq: str
    idx_offset: int
    cron_expr: str
    pdf_parser_type: str
    parent_chunk_size: Optional[int]
    parent_chunk_overlap: Optional[int]
    child_chunk_size: int
    child_chunk_overlap: int
    embedding_model_name: str
    retriever_type: str
    k: int
    multiplier: int
    semantic_weight: float
    use_hyde: bool
    use_rag_fusion: bool
    use_parent_document: bool
    use_time_weight: bool
    use_reorder: bool
    decay_rate: float
    temperature: float


def load_config(config_path: str) -> Config:
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return Config(
        proj_name=get_default(config["proj"]["proj_name"], "wealth-whisper"),
        profile_name=config["proj"]["profile_name"],
        region_name=get_default(config["proj"]["region_name"], "us-east-1"),
        stage=get_default(config["proj"]["stage"], "dev"),
        idx_duration=get_default(config["data"]["idx_duration"], 0),
        idx_freq=get_default(config["data"]["idx_freq"], "d"),
        idx_offset=get_default(config["data"]["idx_offset"], 0),
        cron_expr=get_default(config["data"]["cron_expr"], "0 0 * * ?"),
        pdf_parser_type=get_default(config["data"]["pdf_parser_type"], "pypdf"),
        parent_chunk_size=config["data"]["parent_chunk_size"],
        parent_chunk_overlap=config["data"]["parent_chunk_overlap"],
        child_chunk_size=get_default(config["data"]["child_chunk_size"], 300),
        child_chunk_overlap=get_default(config["data"]["child_chunk_overlap"], 30),
        embedding_model_name=get_default(
            config["retriever"]["embedding_model_name"], "amazon"
        ),
        retriever_type=get_default(config["retriever"]["retriever_type"], "semantic"),
        k=get_default(config["retriever"]["k"], 5),
        multiplier=get_default(config["retriever"]["multiplier"], 1),
        semantic_weight=get_default(config["retriever"]["semantic_weight"], 0.5),
        use_hyde=get_default(config["retriever"]["use_hyde"], False),
        use_rag_fusion=get_default(config["retriever"]["use_rag_fusion"], False),
        use_parent_document=get_default(
            config["retriever"]["use_parent_document"], False
        ),
        use_time_weight=get_default(config["retriever"]["use_time_weight"], False),
        use_reorder=get_default(config["retriever"]["use_reorder"], False),
        decay_rate=get_default(config["retriever"]["decay_rate"], 0.01),
        temperature=get_default(config["llm"]["temperature"], 0.0),
    )
