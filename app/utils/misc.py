import argparse
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import numpy as np
from dateutil.relativedelta import relativedelta


def arg_as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def calc_cos_sim(x_arr: np.array, y_arr: np.array) -> float:
    if len(x_arr.shape) == 1:
        x_arr = x_arr[np.newaxis, :]
    if len(y_arr.shape) == 1:
        y_arr = y_arr[np.newaxis, :]

    x_arr = x_arr / np.linalg.norm(x_arr, axis=1)[:, np.newaxis]
    y_arr = y_arr / np.linalg.norm(y_arr, axis=1)[:, np.newaxis]

    return np.dot(x_arr, y_arr.T)[0][0]


def convert_date_string(date_string: str) -> str:
    return datetime.strptime(date_string, "%Y%m%d").strftime("%Y-%m-%d")


def get_date_values(start_date_string: str, end_date_string: str) -> List[str]:
    start_date = datetime.strptime(start_date_string, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_string, "%Y-%m-%d")

    date_strings = []
    curr_date = start_date
    while curr_date <= end_date:
        date_strings.append(curr_date.strftime("%Y%m%d"))
        curr_date += timedelta(days=1)

    return date_strings


def get_default(value: Optional[Any], default: Any) -> Any:
    return default if value is None else value


def log_or_print(msg: str, logger: Optional[logging.Logger] = None) -> None:
    if logger:
        logger.info(msg)
    else:
        print(msg)


def parse_cron_expr(cron_expr: str) -> Dict[str, str]:
    parts = cron_expr.split()

    if len(parts) < 5 or len(parts) > 6:
        raise ValueError(
            "This is an invalid cron expression. The expected format is 'minute hour day month day [year]'."
        )

    return {
        "minute": parts[0],
        "hour": parts[1],
        "day": parts[2] if parts[2] != "?" else None,
        "month": parts[3] if parts[3] != "?" else None,
        "week_day": parts[4] if parts[4] not in ["?", "*"] else None,
    }


def shift_datetime(
    base: str, shift: int, freq: str = "d", fmt: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    shift_dict = {
        "d": {"days": shift},
        "w": {"weeks": shift},
        "m": {"months": shift},
        "y": {"years": shift},
    }
    delta = shift_dict.get(freq.lower())

    if not delta:
        raise ValueError(f"Invalid frequency: {freq}")

    return (datetime.strptime(base, fmt) + relativedelta(**delta)).strftime(fmt)
