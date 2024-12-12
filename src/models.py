from enum import StrEnum, auto
from typing import Any
from fastapi import Form
from pydantic import BaseModel, ConfigDict, Field


class BaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OutputFormat(StrEnum):
    MARKDOWN = auto()
    TEXT = auto()
    HTML = auto()


class ParseRequest(BaseRequest):
    include_json: bool = Field(
        False,
        description="Include a json representation of the document in the response",
    )
    output_format: OutputFormat = Field(
        OutputFormat.MARKDOWN, description="Output format of parsed text"
    )


class ParseUrlRequest(ParseRequest):
    url: str = Field(..., description="Download url for input file")


class ParseFileRequest(ParseRequest):
    @classmethod
    def from_form_data(
        cls,
        data: str = Form(..., examples=[ParseRequest().model_dump_json()]),
    ) -> "ParseFileRequest":
        return cls.model_validate_json(data)


class BaseResponse(BaseModel):
    message: str
    status: str


class ParseResponseData(BaseModel):
    output: str
    json_output: dict[str, Any] | None = None


class ParseResponse(BaseResponse):
    data: ParseResponseData
