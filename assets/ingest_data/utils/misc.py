import logging
from datetime import datetime, timedelta
from typing import Any, List, Optional
from dateutil.relativedelta import relativedelta


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
