import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from .aws_helpers import (
    create_opensearch_client,
    download_dir_from_s3,
    download_file_from_s3,
    get_opensearch_domain_endpoint,
    get_secret,
    get_ssm_param_value,
    make_s3_uri,
)
from .common import get_dimensions, get_model_id, get_name, get_ssm_param_key
from .config_handler import load_config
from .logger import logger
from .misc import (
    get_date_values,
    get_default,
    log_or_print,
    shift_datetime,
)
from .rag_utils import describe_docs_stat, get_embeddings, save_docs_to_jsonl, show_docs
