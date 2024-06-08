import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from app.utils import get_model_id, get_name, get_retriever, load_config, logger


def get_dir_path(dir_name: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), dir_name))


if __name__ == "__main__":
    config_dir = get_dir_path(os.path.join(os.pardir, "app", "configs"))
    config = load_config(os.path.join(config_dir, "config.yaml"))

    retriever = get_retriever(
        config.profile_name,
        config.region_name,
        get_name(config.proj_name, config.stage, "docs"),
        get_name(config.proj_name, config.stage, "docs"),
        get_name(config.proj_name, config.stage, "embedding"),
        get_name(config.proj_name, config.stage, "docs"),
        get_model_id(config.embedding_model_name),
        config.retriever_type,
        config.k,
        config.multiplier,
        config.semantic_weight,
        config.use_hyde,
        config.use_rag_fusion,
        config.use_parent_document,
        config.use_time_weight,
        config.use_reorder,
        config.decay_rate,
        config.temperature,
        verbose=True,
        logger=logger,
    )

    # Add your test samples
    queries = ["삼성전자 예상 실적은?", "미국 금리 전망은?"]

    for query in queries:
        results = retriever.get_relevant_documents(query)
        string = "-" * 100 + f"\nquestion: {query}\nanswer:"
        for i, result in enumerate(results[: config.k]):
            string += f"\n{i + 1}: {result.page_content.strip()}"
        logger.info(string)
