from enum import StrEnum, auto
from typing import Any, Dict, List
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
    chunk_document: bool = Field(
        False, 
        description="Process the document with Hybrid Chunker and return chunks"
    )
    max_tokens_per_chunk: int = Field(
        512, 
        description="Maximum tokens per chunk when chunking is enabled"
    )
    optimize_pdf: bool = Field(
        True,
        description="Apply PDF optimization before parsing (improves text extraction)"
    )
    merge_sections: bool = Field(
        True,
        description="Merge chunks with the same section title across page boundaries"
    )


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


class ChunkData(BaseModel):
    """Data for a single chunk from the document"""
    section_title: str | None = None
    text: str
    chunk_index: int
    metadata: Dict[str, Any]


class ChunkResponseData(BaseModel):
    """Response data when chunking is enabled"""
    chunks: List[ChunkData]
    json_output: dict[str, Any] | None = None
    
    class Config:
        # Exclude null values from the response
        extra = "ignore" 
        json_encoders = {
            # Custom encoders here
        }
        
        @staticmethod
        def schema_extra(schema: Dict[str, Any]) -> None:
            for prop in schema.get("properties", {}).values():
                prop.pop("nullable", None)
                
        @classmethod
        def _exclude_none(cls, data: Dict[str, Any]) -> Dict[str, Any]:
            return {k: v for k, v in data.items() if v is not None}


class ParseResponse(BaseResponse):
    data: ParseResponseData | ChunkResponseData
