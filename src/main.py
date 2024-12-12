from pathlib import Path
from contextlib import asynccontextmanager
from io import BytesIO
from typing import AsyncIterator, Callable
import logging

from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    InputFormat,
)
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.io import DocumentStream
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import uvicorn


from src.models import (
    OutputFormat,
    ParseFileRequest,
    ParseResponse,
    ParseResponseData,
    ParseUrlRequest,
)
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

    app.state.config = Config()

    yield
    # Teardown


app = FastAPI(lifespan=lifespan)

bearer_auth = HTTPBearer(auto_error=False)


async def authorize_header(
    request: Request, bearer: HTTPAuthorizationCredentials | None = Depends(bearer_auth)
):
    # Do nothing if AUTH_KEY is not set
    auth_token: str | None = request.app.state.config.auth_token
    if auth_token is None:
        return

    # Validate auth bearer
    if bearer is None or bearer.credentials != auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": "Unauthorized"},
        )


@app.exception_handler(Exception)
async def ingestion_error_handler(_, exc: Exception):
    detail = {"message": str(exc)}
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail
    )


ConvertData = str | Path | DocumentStream
ConvertFunc = Callable[[ConvertData], ConversionResult]


def convert(request: Request) -> ConvertFunc:
    def convert_func(data: ConvertData) -> ConversionResult:
        try:
            result = request.app.state.converter.convert(data, raises_on_error=False)
            _check_conversion_result(result)
            return result
        except FileNotFoundError as exc:
            logger.error(f"File not found error: {str(exc)}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "Input not found"},
            ) from exc

    return convert_func


@app.post("/parse/url")
def parse_document_url(
    payload: ParseUrlRequest,
    convert: ConvertFunc = Depends(convert),
    _=Depends(authorize_header),
):
    result = convert(payload.url)
    output = _get_output(result.document, payload.output_format)

    json_output = result.document.export_to_dict() if payload.include_json else None

    return ParseResponse(
        message="Document parsed successfully",
        status="Ok",
        data=ParseResponseData(output=output, json_output=json_output),
    )


@app.post("/parse/file")
def parse_document_stream(
    file: UploadFile,
    convert: ConvertFunc = Depends(convert),
    payload: ParseFileRequest = Depends(ParseFileRequest.from_form_data),
    _=Depends(authorize_header),
):
    binary_data = file.file.read()
    data = DocumentStream(
        name=file.filename or "unset_name", stream=BytesIO(binary_data)
    )

    result = convert(data)
    output = _get_output(result.document, payload.output_format)

    json_output = result.document.export_to_dict() if payload.include_json else None

    return ParseResponse(
        message="Document parsed successfully",
        status="Ok",
        data=ParseResponseData(output=output, json_output=json_output),
    )


def _check_conversion_result(result: ConversionResult) -> None:
    """Raises HTTPException and logs on error"""
    if result.status in [ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS]:
        return

    if result.errors:
        for error in result.errors:
            if error.component_type == DoclingComponentType.USER_INPUT:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={"message": error.error_message},
                )
            logger.error(
                f"Error in: {error.component_type.name} - {error.error_message}"
            )
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


def _get_output(document: DoclingDocument, format: OutputFormat) -> str:
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
