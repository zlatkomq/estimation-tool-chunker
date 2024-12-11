from enum import StrEnum, auto
from pydantic import BaseModel


class OutputFormat(StrEnum):
    MARKDOWN = auto()
    TEXT = auto()
    HTML = auto()


class BaseResponse(BaseModel):
    message: str
    status: str


class ParseResponseData(BaseModel):
    output: str


class ParseResponse(BaseResponse):
    data: ParseResponseData
