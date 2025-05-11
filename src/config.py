from pathlib import Path
import multiprocessing

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    num_workers: int | float | None = None
    log_level: str = "INFO"
    dev_mode: bool = False
    port: int = 8080
    auth_token: str | None = None
    ocr_languages: str = "en,es,fr,de,sv"
    do_code_enrichment: bool = False
    do_formula_enrichment: bool = False
    do_picture_classification: bool = False
    do_picture_description: bool = False

    def get_num_workers(self) -> int | None:
        if self.num_workers is None:
            return None

        if self.num_workers == -1:
            return multiprocessing.cpu_count()

        if 0 < self.num_workers < 1:
            return int(self.num_workers * multiprocessing.cpu_count())

        return int(self.num_workers)


def get_log_config(log_level: str):
    log_file = Path.cwd() / Path("logs/docling-inference.log")
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
                "level": log_level,
            },
            "uvicorn": {
                "handlers": ["uvicorn"],
                "level": log_level,  # Use the same log level as the main app
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["uvicorn"],
                "level": log_level,  # Use the same log level as the main app
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["uvicorn", "file"],  # Add file handler for access logs
                "level": log_level,  # Use the same log level as the main app
                "propagate": False,
            },
            "src": {  # Our app module logger
                "handlers": ["default", "file"],
                "level": log_level,
                "propagate": False,
            },
            "docling": {
                "handlers": ["default", "file"],
                "level": log_level,  # Use the same log level as the main app
                "propagate": False,
            },
        },
    }
