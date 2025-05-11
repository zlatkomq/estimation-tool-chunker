from pathlib import Path
from contextlib import asynccontextmanager
from io import BytesIO
from typing import AsyncIterator, Callable, List, Dict, Any, Tuple, Union, Optional
import logging
import logging.config
import tempfile
import os
import re
import time

from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    InputFormat,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.io import DocumentStream
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker import BaseChunk
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import JSONResponse
import uvicorn

# Import for Nomic tokenizer if available
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False

# Import for PDF optimization
try:
    import pikepdf
    from pikepdf import Pdf
    PDF_OPTIMIZATION_AVAILABLE = True
except ImportError:
    PDF_OPTIMIZATION_AVAILABLE = False


from src.models import (
    OutputFormat,
    ParseFileRequest,
    ParseResponse,
    ParseResponseData,
    ChunkData,
    ChunkResponseData,
)
from src.config import Config, get_log_config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Setup and teardown events of the app"""
    # Setup
    logger.info("Starting application initialization")
    config = Config()
    logger.info(f"Configuration loaded: OCR languages: {config.ocr_languages}, Workers: {config.workers}")

    ocr_languages = config.ocr_languages.split(",")
    logger.info(f"Initializing DocumentConverter with OCR languages: {ocr_languages}")
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(
                    ocr_options=EasyOcrOptions(lang=ocr_languages),
                    do_code_enrichment=False,
                    do_formula_enrichment=False,
                    do_picture_classification=False,
                    do_picture_description=False,
                )
            )
        }
    )

    # Since we only work with PDFs, only initialize the PDF pipeline
    logger.info("Initializing PDF pipeline only")
    converter.initialize_pipeline(InputFormat.PDF)
    logger.info("PDF pipeline initialization complete")

    app.state.converter = converter
    app.state.config = config

    # Initialize Nomic tokenizer if available
    if TOKENIZER_AVAILABLE:
        try:
            logger.info("Attempting to load Nomic tokenizer")
            app.state.tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v2-moe")
            logger.info("Nomic tokenizer initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to load Nomic tokenizer: {str(e)}")
            app.state.tokenizer = None
    else:
        logger.warning("Transformers library not available, Nomic tokenizer will not be used")
        app.state.tokenizer = None

    logger.info("Application initialization complete")
    yield
    # Teardown
    logger.info("Application shutting down")


app = FastAPI(
    lifespan=lifespan,
    default_response_class=JSONResponse
)

bearer_auth = HTTPBearer(auto_error=False)


async def authorize_header(
    request: Request, bearer: HTTPAuthorizationCredentials | None = Depends(bearer_auth)
) -> None:
    # Do nothing if AUTH_KEY is not set
    auth_token: str | None = request.app.state.config.auth_token
    if auth_token is None:
        logger.debug("No authentication token configured, skipping authorization")
        return

    # Validate auth bearer
    if bearer is None or bearer.credentials != auth_token:
        logger.warning("Failed authentication attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": "Unauthorized"},
        )
    logger.debug("Authentication successful")


@app.exception_handler(Exception)
async def ingestion_error_handler(_, exc: Exception) -> None:
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    detail = {"message": str(exc)}
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail
    )


ConvertData = Path | DocumentStream
ConvertFunc = Callable[[ConvertData], ConversionResult]


def convert(request: Request) -> ConvertFunc:
    def convert_func(data: ConvertData) -> ConversionResult:
        try:
            logger.info("Starting document conversion")
            start_time = time.time()
            result = request.app.state.converter.convert(data, raises_on_error=False)
            conversion_time = time.time() - start_time
            logger.info(f"Document conversion completed in {conversion_time:.2f} seconds")
            _check_conversion_result(result)
            return result
        except FileNotFoundError as exc:
            logger.error(f"File not found error: {str(exc)}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "Input not found"},
            ) from exc

    return convert_func


def get_heading_path(headings):
    """Get the full heading path as a breadcrumb"""
    path = " > ".join(headings) if headings else None
    logger.debug(f"Extracted heading path: {path}")
    return path

def get_content_type(chunk):
    """Determine content type based on doc items"""
    types = set()
    if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
        for item in chunk.meta.doc_items:
            if hasattr(item, "label"):
                types.add(item.label)
    
    if "table" in types:
        content_type = "table"
    elif "list_item" in types:
        content_type = "list"
    elif "section_header" in types:
        content_type = "heading"
    else:
        content_type = "text"
    
    logger.debug(f"Determined content type '{content_type}' from types: {types}")
    return content_type

def get_page_range(pages):
    """Format page range nicely"""
    if not pages:
        return None
    if len(pages) == 1:
        result = f"Page {pages[0]}"
    else:
        result = f"Pages {min(pages)}-{max(pages)}"
    logger.debug(f"Page range determined: {result} from pages: {pages}")
    return result

def extract_table_data(chunk):
    """Extract structured table data if present"""
    if not hasattr(chunk.meta, "doc_items") or not chunk.meta.doc_items:
        return None
    
    table_data = []
    for item in chunk.meta.doc_items:
        if hasattr(item, "label") and item.label == "table" and hasattr(item, "data"):
            if hasattr(item.data, "table_cells") and item.data.table_cells:
                rows = []
                # Try to reconstruct table in a readable format
                for cell in item.data.table_cells:
                    if hasattr(cell, "text") and cell.text:
                        rows.append(f"{cell.text}")
                if rows:
                    table_data.append(" | ".join(rows))
    
    if table_data:
        logger.debug(f"Extracted table data with {len(table_data)} rows")
    return table_data if table_data else None

def get_list_info(chunk):
    """Extract list information if present"""
    if not hasattr(chunk.meta, "doc_items") or not chunk.meta.doc_items:
        return None
    
    list_items = []
    for item in chunk.meta.doc_items:
        if hasattr(item, "label") and item.label == "list_item":
            marker = getattr(item, "marker", "-")
            enumerated = getattr(item, "enumerated", False)
            text = getattr(item, "text", "")
            list_items.append({
                "marker": marker,
                "enumerated": enumerated,
                "text": text
            })
    
    if list_items:
        logger.debug(f"Extracted list information with {len(list_items)} items")
    return list_items if list_items else None

def get_document_summary(document):
    """Extract document-level summary information"""
    summary = {}
    
    if hasattr(document, "name") and document.name:
        summary["name"] = document.name
        
    if hasattr(document, "pages"):
        summary["total_pages"] = len(document.pages)
    
    if summary:
        logger.debug(f"Document summary: {summary}")
    return summary if summary else None

def extract_captions(chunk):
    """Extract captions from tables, figures, and other elements"""
    captions = []
    
    # Check if doc_items exist
    if not hasattr(chunk.meta, "doc_items") or not chunk.meta.doc_items:
        return captions
    
    # Extract captions from different elements
    for item in chunk.meta.doc_items:
        # Get captions from tables
        if hasattr(item, "label") and item.label == "table":
            if hasattr(item, "captions") and item.captions:
                captions.extend(item.captions)
    
    if captions:
        logger.debug(f"Extracted {len(captions)} captions: {captions}")
    return captions

class TokenizerAdapter:
    """Adapter to provide a consistent interface for different tokenizer types"""
    
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the appropriate method for the tokenizer type"""
        try:
            # Check if the tokenizer already has count_tokens method (Docling interface)
            if hasattr(self.tokenizer, "count_tokens"):
                return self.tokenizer.count_tokens(text)
            
            # For Hugging Face tokenizers
            elif hasattr(self.tokenizer, "encode"):
                return len(self.tokenizer.encode(text))
                
            # For Transformers tokenizers that use __call__
            elif callable(self.tokenizer):
                return len(self.tokenizer(text)["input_ids"])
                
            # Fallback to rough estimation
            else:
                logger.warning("Tokenizer doesn't have compatible interface, using word estimation")
                return self._estimate_tokens(text)
                
        except Exception as e:
            logger.warning(f"Error counting tokens: {str(e)}. Using word estimation.")
            return self._estimate_tokens(text)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count based on whitespace (less accurate)"""
        # Apply a factor for safety since words usually tokenize to more than one token
        return int(len(re.findall(r'\S+', text)) * 1.3)
    
    @staticmethod
    def create(tokenizer: Optional[Any] = None) -> 'TokenizerAdapter':
        """Create a tokenizer adapter from a tokenizer object or a default estimator if None"""
        if tokenizer is None:
            # Create a null tokenizer that just estimates
            return TokenizerAdapter(None)
        return TokenizerAdapter(tokenizer)


def merge_chunks_by_section(chunks: List[ChunkData], max_tokens: int, tokenizer=None) -> List[ChunkData]:
    """Merge chunks that have the same section title, respecting token limit."""
    logger.info(f"Merging chunks by section (max tokens: {max_tokens})")
    start_time = time.time()
    
    if not chunks:
        return []
    
    # Create a consistent tokenizer interface if not already an adapter
    if not isinstance(tokenizer, TokenizerAdapter):
        token_counter = TokenizerAdapter.create(tokenizer)
    else:
        token_counter = tokenizer
    
    # Group chunks by section title
    sections: Dict[str, List[Tuple[int, ChunkData]]] = {}
    for idx, chunk in enumerate(chunks):
        key = chunk.section_title or ""
        if key not in sections:
            sections[key] = []
        sections[key].append((idx, chunk))
    
    # Prepare merged chunks list
    merged_chunks: List[ChunkData] = []
    used_indices = set()
    
    # For each section, try to merge its chunks
    for section_title, section_chunks in sections.items():
        if len(section_chunks) <= 1:
            # Only one chunk in this section, no need to merge
            merged_chunks.append(section_chunks[0][1])
            used_indices.add(section_chunks[0][0])
            continue
        
        # Sort chunks by original index to maintain document order
        section_chunks.sort(key=lambda x: x[0])
        
        # Process each chunk in order
        current_merged_chunk = None
        current_indices = []
        
        for idx, chunk in section_chunks:
            if not current_merged_chunk:
                # Start a new merged chunk
                current_merged_chunk = chunk
                current_indices = [idx]
                continue
            
            # Try to merge with current_merged_chunk
            candidate_text = current_merged_chunk.text + "\n" + chunk.text
            
            # Count tokens using our adapter
            token_count = token_counter.count_tokens(candidate_text)
            
            if token_count <= max_tokens:
                # Safe to merge
                # Merge text content
                current_merged_chunk.text = candidate_text
                
                # Merge page numbers in metadata
                current_pages = set(current_merged_chunk.metadata.get("pages", []))
                chunk_pages = set(chunk.metadata.get("pages", []))
                if current_pages or chunk_pages:
                    current_merged_chunk.metadata["pages"] = sorted(list(current_pages.union(chunk_pages)))
                
                # Keep the earliest chunk_index from the merged chunks
                current_merged_chunk.chunk_index = min(current_merged_chunk.chunk_index, chunk.chunk_index)
                
                # Add this index to current group
                current_indices.append(idx)
            else:
                # Would exceed token limit, finish current merge group
                merged_chunks.append(current_merged_chunk)
                for i in current_indices:
                    used_indices.add(i)
                
                # Start a new merge group with this chunk
                current_merged_chunk = chunk
                current_indices = [idx]
        
        # Add the last merge group if it exists
        if current_merged_chunk:
            merged_chunks.append(current_merged_chunk)
            for i in current_indices:
                used_indices.add(i)
    
    # Add any chunks that weren't merged (different section titles or no title)
    for idx, chunk in enumerate(chunks):
        if idx not in used_indices:
            merged_chunks.append(chunk)
    
    # Sort merged chunks by their position in the original document
    # Find the minimum original index for each section
    section_min_indices = {}
    for chunk in merged_chunks:
        section_title = chunk.section_title or ""
        if section_title not in section_min_indices:
            # Find the earliest index in the original chunks list with this section title
            matching_indices = [i for i, c in enumerate(chunks) 
                              if c.section_title == chunk.section_title]
            if matching_indices:
                section_min_indices[section_title] = min(matching_indices)
            else:
                # Fallback position if no matching chunks found (shouldn't happen)
                section_min_indices[section_title] = len(chunks)
    
    # Sort by the earliest position the section appears in the original document
    merged_chunks.sort(key=lambda chunk: section_min_indices.get(chunk.section_title or "", len(chunks)))
    
    merge_time = time.time() - start_time
    logger.info(f"Chunk merging completed in {merge_time:.2f} seconds. Reduced from {len(chunks)} to {len(merged_chunks)} chunks")
    return merged_chunks

def chunk_document(document: DoclingDocument, max_tokens: int, tokenizer=None, merge_sections=True) -> List[ChunkData]:
    """Process document into semantic chunks with metadata."""
    logger.info(f"Chunking document, max_tokens={max_tokens}, merge_sections={merge_sections}")
    start_time = time.time()
    
    # Initialize tokenizer adapter
    token_counter = TokenizerAdapter.create(tokenizer)
    logger.debug(f"Using tokenizer: {token_counter.__class__.__name__}")
    
    # Initialize chunker
    chunker = HybridChunker(max_tokens=max_tokens, tokenizer=token_counter)
    logger.debug("Hybrid chunker initialized")
    
    chunks = []
    
    # Process document with HybridChunker
    for chunk_idx, chunk in enumerate(chunker.chunk(document)):
        # Add contextualization to each chunk
        contextualized_text = chunker.contextualize(chunk)
        
        # Add Nomic prefix for optimal embedding
        prefixed_text = f"search_document: {contextualized_text}"
        
        # Extract the section title from headings
        section_title = None
        if hasattr(chunk.meta, "headings") and chunk.meta.headings:
            # Use the most specific (last) heading
            section_title = chunk.meta.headings[-1]
        
        # Get page numbers if available
        page_numbers = set()
        if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
            for item in chunk.meta.doc_items:
                if hasattr(item, "prov"):
                    for prov in item.prov:
                        if hasattr(prov, "page_no") and prov.page_no is not None:
                            page_numbers.add(prov.page_no)
        
        # Extract captions properly
        extracted_captions = extract_captions(chunk)
        
        # Create enhanced metadata that will be useful for embedding and retrieval
        metadata = {
            "content_type": get_content_type(chunk),
            "heading_path": get_heading_path(getattr(chunk.meta, "headings", [])),
        }
        
        # Only add captions if they exist
        if extracted_captions:
            metadata["captions"] = extracted_captions
        
        # Add document path if available
        if hasattr(document, "path") and document.path:
            metadata["source_path"] = str(document.path)
            
        # Add page numbers if available
        if page_numbers:
            metadata["pages"] = sorted(list(page_numbers))
        
        # Extract and add table data if present
        table_data = extract_table_data(chunk)
        if table_data:
            metadata["table_data"] = table_data
        
        # Handle origin data but filter out unwanted fields
        if hasattr(chunk.meta, "origin") and chunk.meta.origin:
            # Create a filtered copy of origin without the unwanted fields
            origin = {}
            if isinstance(chunk.meta.origin, dict):
                for key, value in chunk.meta.origin.items():
                    if key not in ["mimetype", "binary_hash", "filename", "uri", "name"]:
                        origin[key] = value
                if origin:  # Only add if there are remaining fields
                    metadata["origin"] = origin
        
        # Convert BaseChunk to ChunkData
        chunk_data = ChunkData(
            text=prefixed_text,
            section_title=section_title,
            chunk_index=chunk_idx,
            metadata=metadata
        )
        chunks.append(chunk_data)
    
    # Apply section merging if requested
    if merge_sections:
        logger.info("Applying section merging to chunks")
        chunks = merge_chunks_by_section(chunks, max_tokens, tokenizer)
    
    # Add sequential indices to chunks
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i
    
    chunking_time = time.time() - start_time
    logger.info(f"Document chunking completed in {chunking_time:.2f} seconds. Generated {len(chunks)} chunks")
    return chunks


def optimize_pdf(binary_data: bytes) -> bytes:
    """Optimize PDF for better text extraction."""
    if not PDF_OPTIMIZATION_AVAILABLE:
        logger.warning("PDF optimization requested but pikepdf is not available. Using original PDF.")
        return binary_data
    
    logger.info("Starting PDF optimization")
    start_time = time.time()
    
    try:
        # Create a temporary file to work with the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_in, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_out:
            
            # Write the input binary data to the temp file
            tmp_in.write(binary_data)
            tmp_in.flush()
            
            # Open the PDF with pikepdf
            with Pdf.open(tmp_in.name) as pdf:
                # Remove unnecessary metadata that might interfere with parsing
                pdf.remove_unreferenced_resources()
                
                # Save with optimization options - can't use both normalize_content and linearize together
                pdf.save(tmp_out.name, 
                         linearize=False,  # Changed to False to avoid conflict with normalize_content
                         object_stream_mode=pikepdf.ObjectStreamMode.generate,
                         compress_streams=True, 
                         normalize_content=True,
                         qdf=False)
            
            # Read the optimized PDF
            with open(tmp_out.name, 'rb') as f:
                optimized_data = f.read()
                
            # Clean up temporary files
            os.unlink(tmp_in.name)
            os.unlink(tmp_out.name)
            
            logger.info("PDF successfully optimized")
            return optimized_data
            
    except Exception as e:
        logger.error(f"Error during PDF optimization: {str(e)}. Using original PDF.")
        return binary_data
    finally:
        optimization_time = time.time() - start_time
        logger.info(f"PDF optimization completed in {optimization_time:.2f} seconds")


@app.post("/parse/file", response_model=ParseResponse)
def parse_document_stream(
    file: UploadFile,
    convert: ConvertFunc = Depends(convert),
    payload: ParseFileRequest = Depends(ParseFileRequest.from_form_data),
    request: Request = None,
    _=Depends(authorize_header),
) -> ParseResponse:
    # Validate that the file is a PDF
    if file.content_type != "application/pdf":
        logger.warning(f"Unsupported file type: {file.content_type}. Only PDFs are accepted.")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"message": "Only PDF files are supported"},
        )
    
    logger.info(f"Processing file: {file.filename}, size: {file.size} bytes")
    logger.debug(f"Request parameters: {payload.dict()}")
    
    # Read the file content
    binary_data = file.file.read()
    logger.debug(f"File read completed, size: {len(binary_data)} bytes")
    
    # Optimize PDF if requested
    if payload.optimize_pdf:
        logger.info("PDF optimization requested")
        try:
            binary_data = optimize_pdf(binary_data)
            logger.info("PDF successfully optimized")
        except Exception as e:
            logger.error(f"PDF optimization failed: {str(e)}", exc_info=True)
            logger.info("Proceeding with original PDF")
    
    # Create temporary file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(binary_data)
        tmp_path = Path(tmp.name)
    
    logger.debug(f"Temporary file created at {tmp_path}")
    
    try:
        # Process the document
        start_time = time.time()
        result = convert(tmp_path)
        document = result.document
        logger.info(f"Document converted successfully in {time.time() - start_time:.2f} seconds")
        
        # Process according to request parameters
        if payload.chunk_document:
            logger.info("Processing document with HybridChunker")
            chunks = chunk_document(
                document,
                payload.max_tokens_per_chunk,
                request.app.state.tokenizer if request and hasattr(request.app.state, "tokenizer") else None,
                payload.merge_sections
            )
            
            # Create response
            response_data = ChunkResponseData(chunks=chunks)
            logger.info(f"Returning {len(chunks)} chunks")
            return ParseResponse(data=response_data)
        else:
            # Return document in requested format
            logger.info(f"Converting document to {payload.output_format} format")
            output_text = _get_output(document, payload.output_format)
            
            # Create response
            response_data = ParseResponseData(
                document=output_text,
                json=document.to_dict() if payload.include_json else None,
            )
            logger.info(f"Returning document with {len(output_text)} characters")
            return ParseResponse(data=response_data)
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up temporary file
        try:
            if tmp_path.exists():
                os.unlink(tmp_path)
                logger.debug(f"Temporary file {tmp_path} deleted")
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {tmp_path}: {str(e)}")


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


# Custom JSON response class to exclude nulls 
@app.middleware("http")
async def exclude_nulls_middleware(request: Request, call_next):
    response = await call_next(request)
    
    if isinstance(response, JSONResponse):
        response_body = response.body.decode("utf-8")
        
        # Only process JSON responses
        try:
            import json
            import re
            
            # Remove null fields from the JSON
            cleaned_body = re.sub(r'"(json_output|document)":null,?', '', response_body)
            
            # Create new response without null fields
            return JSONResponse(
                status_code=response.status_code,
                content=json.loads(cleaned_body),
                headers=dict(response.headers),
                media_type=response.media_type,
            )
        except:
            # If any error occurs, return the original response
            return response
    
    return response


if __name__ == "__main__":
    config = Config()
    # Always set log level explicitly to ensure logs are visible in all modes
    log_level = config.log_level
    if log_level == "INFO" and not config.dev_mode:
        # Ensure we see important logs in production mode
        log_level = "INFO"
    
    # Ensure Python output is not buffered which can hide logs in Docker
    if os.environ.get("PYTHONUNBUFFERED", "0") != "1":
        os.environ["PYTHONUNBUFFERED"] = "1"
        logger.warning("PYTHONUNBUFFERED was not set. Setting it to 1 to ensure logs are visible.")
    
    # Initialize logging before app starts
    logging.config.dictConfig(get_log_config(log_level))
    logger.info(f"Starting application with LOG_LEVEL={log_level}, DEV_MODE={config.dev_mode}, WORKERS={config.get_num_workers()}")
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=config.port,
        log_config=get_log_config(log_level),
        reload=config.dev_mode,
        workers=config.get_num_workers(),
    )
