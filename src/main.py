from pathlib import Path
from contextlib import asynccontextmanager
from io import BytesIO
from typing import AsyncIterator, Callable
import logging

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter
from docling.exceptions import ConversionError
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.io import DocumentStream
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Path as PathParam,
    Request,
    UploadFile,
    status,
)
import uvicorn


from src.models import OutputFormat, ParseResponse, ParseResponseData
from src.config import Config, get_log_config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Setup and teardown events of the app"""
    # Setup
    converter = DocumentConverter()
    for i, format in enumerate(InputFormat):
        logger.info(f"Initializing {format.value} pipeline {i + 1}/{len(InputFormat)}")
        converter.initialize_pipeline(format)
    app.state.converter = converter

    yield
    # Teardown


app = FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
async def ingestion_error_handler(_, exc: Exception):
    detail = {"message": str(exc)}
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail
    )


ConvertFunc = Callable[[str | Path | DocumentStream], ConversionResult]


def converter(request: Request) -> ConvertFunc:
    def convert_func(data: str | Path | DocumentStream):
        try:
            return request.app.state.converter.convert(data)
        except ConversionError as exc:
            if str(exc).startswith("File format not allowed"):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={"message": "File format not allowed"},
                ) from exc
            if "No such file or directory" in str(exc):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"message": "Document not found"},
                ) from exc

            raise

    return convert_func


@app.post("/parse/{url}")
def parse_document_url(
    url: str = PathParam(..., description="Download url of document"),
    format: OutputFormat = OutputFormat.MARKDOWN,
    converter: ConvertFunc = Depends(converter),
):
    result = converter(url)
    output = _convert_document(result.document, format)

    return ParseResponse(
        message="Document parsed successfully",
        status="Ok",
        data=ParseResponseData(output=output),
    )


@app.post("/parse")
def parse_document_stream(
    file: UploadFile,
    format: OutputFormat = OutputFormat.MARKDOWN,
    converter: ConvertFunc = Depends(converter),
):
    binary_data = file.file.read()
    data = DocumentStream(
        name=file.filename or "unset_name", stream=BytesIO(binary_data)
    )

    result = converter(data)
    output = _convert_document(result.document, format)

    return ParseResponse(
        message="Document parsed successfully",
        status="Ok",
        data=ParseResponseData(output=output),
    )


def _convert_document(document: DoclingDocument, format: OutputFormat) -> str:
    if format == OutputFormat.MARKDOWN:
        return document.export_to_markdown()
    if format == OutputFormat.TEXT:
        return document.export_to_text()
    if format == OutputFormat.HTML:
        return document.export_to_html()


if __name__ == "__main__":
    config = Config()
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=config.port,
        log_config=get_log_config(config.log_level),
        reload=config.dev_mode,
        workers=config.get_num_workers(),
    )
