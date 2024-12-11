import os
from pathlib import Path
import multiprocessing

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    num_workers: int | None = None
    log_level: str = "INFO"
    dev_mode: bool = False


def get_dev_mode() -> bool:
    return bool(os.getenv("DEV_MODE"))


def get_num_workers() -> int | None:
    n_workers_var = os.getenv("NUM_WORKERS")
    if n_workers_var is None:
        return None

    if n_workers_var == "-1":
        return multiprocessing.cpu_count()

    n_workers = float(n_workers_var)
    if 0 < n_workers < 1:
        return int(n_workers * multiprocessing.cpu_count())

    return int(n_workers)


def get_log_config():
    log_file = Path("/logs/docling-inference.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "uvicorn": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "formatter": "default",
                "class": "logging.FileHandler",
                "filename": str(log_file),
                "mode": "a",
            },
            "uvicorn": {
                "formatter": "uvicorn",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["default", "file"],
                "level": os.environ.get("LOG_LEVEL", "INFO"),
            },
            "uvicorn": {
                "handlers": ["uvicorn"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["uvicorn"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["uvicorn"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }
