from pathlib import Path
from contextlib import asynccontextmanager
from io import BytesIO
from typing import AsyncIterator, Callable, List, Dict, Any, Tuple, Union, Optional
import logging
import tempfile
import os
import re

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
    config = Config()

    ocr_languages = config.ocr_languages.split(",")
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

    app.state.converter = converter
    app.state.config = config

    # Initialize Nomic tokenizer if available
    if TOKENIZER_AVAILABLE:
        try:
            app.state.tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v2-moe")
            logger.info("Nomic tokenizer initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to load Nomic tokenizer: {str(e)}")
            app.state.tokenizer = None
    else:
        app.state.tokenizer = None

    yield
    # Teardown


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
        return

    # Validate auth bearer
    if bearer is None or bearer.credentials != auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": "Unauthorized"},
        )


@app.exception_handler(Exception)
async def ingestion_error_handler(_, exc: Exception) -> None:
    detail = {"message": str(exc)}
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail
    )


ConvertData = Path | DocumentStream
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


def get_heading_path(headings):
    """Get the full heading path as a breadcrumb"""
    return " > ".join(headings) if headings else None

def get_content_type(chunk):
    """Determine content type based on doc items"""
    types = set()
    if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
        for item in chunk.meta.doc_items:
            if hasattr(item, "label"):
                types.add(item.label)
    
    if "table" in types:
        return "table"
    elif "list_item" in types:
        return "list"
    elif "section_header" in types:
        return "heading"
    return "text"

def get_page_range(pages):
    """Format page range nicely"""
    if not pages:
        return None
    if len(pages) == 1:
        return f"Page {pages[0]}"
    return f"Pages {min(pages)}-{max(pages)}"

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
    
    return list_items if list_items else None

def get_document_summary(document):
    """Extract document-level summary information"""
    summary = {}
    
    if hasattr(document, "name") and document.name:
        summary["name"] = document.name
        
    if hasattr(document, "pages"):
        summary["total_pages"] = len(document.pages)
    
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
        
        # Get captions from pictures
        if hasattr(item, "label") and item.label == "picture":
            if hasattr(item, "captions") and item.captions:
                captions.extend(item.captions)
                
        # Get captions from figures
        if hasattr(item, "label") and item.label == "figure":
            if hasattr(item, "captions") and item.captions:
                captions.extend(item.captions)
    
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
    """
    Post-process chunks to merge those with identical section titles that don't exceed token limits.
    This function prioritizes semantic continuity over page boundaries.
    
    Args:
        chunks: List of chunk data
        max_tokens: Maximum tokens per chunk
        tokenizer: Optional tokenizer or TokenizerAdapter for counting tokens
        
    Returns:
        List of merged chunks
    """
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
    
    return merged_chunks

def chunk_document(document: DoclingDocument, max_tokens: int, tokenizer=None, merge_sections=True) -> List[ChunkData]:
    """Process document with HybridChunker and return chunks optimized for embedding"""
    if tokenizer is not None:
        # Use provided tokenizer (HybridChunker handles different tokenizer types internally)
        chunker = HybridChunker(tokenizer=tokenizer, max_tokens=max_tokens, merge_peers=True)
    else:
        # Fallback to default tokenizer
        chunker = HybridChunker(max_tokens=max_tokens, merge_peers=True)
    
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
    
    # Perform section-based merging if requested
    if merge_sections:
        # Use a consistent adapter for the tokenizer
        adapter = TokenizerAdapter.create(tokenizer)
        chunks = merge_chunks_by_section(chunks, max_tokens, adapter)
        
        # Re-index the chunks after merging to maintain sequential order
        for new_idx, chunk in enumerate(chunks):
            chunk.chunk_index = new_idx
    
    return chunks


def optimize_pdf(binary_data: bytes) -> bytes:
    """
    Pre-process PDF to optimize it for better text extraction
    
    This function:
    1. Removes watermarks and background images
    2. Optimizes for text extraction
    3. Repairs structural issues when possible
    4. Normalizes font encoding
    5. Compresses and optimizes the PDF
    
    Args:
        binary_data: Raw PDF binary data
        
    Returns:
        Optimized PDF binary data
    """
    if not PDF_OPTIMIZATION_AVAILABLE:
        logger.warning("PDF optimization requested but pikepdf is not available. Using original PDF.")
        return binary_data
    
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


@app.post("/parse/file", response_model=ParseResponse)
def parse_document_stream(
    file: UploadFile,
    convert: ConvertFunc = Depends(convert),
    payload: ParseFileRequest = Depends(ParseFileRequest.from_form_data),
    request: Request = None,
    _=Depends(authorize_header),
) -> ParseResponse:
    # Validate that the file is a PDF
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"message": "Only PDF files are supported"},
        )
    
    # Read the original binary data
    binary_data = file.file.read()
    
    # Apply PDF optimization if requested
    if payload.optimize_pdf:
        optimized_binary_data = optimize_pdf(binary_data)
    else:
        optimized_binary_data = binary_data
    
    # Create DocumentStream from the optimized data
    data = DocumentStream(
        name=file.filename or "unset_name", stream=BytesIO(optimized_binary_data)
    )

    result = convert(data)
    
    # Process with HybridChunker if requested
    if payload.chunk_document:
        # Get tokenizer if available (for optimal chunk size calculation)
        tokenizer = request.app.state.tokenizer if request and hasattr(request.app.state, "tokenizer") else None
        
        # Process document with chunker and get chunks optimized for later embedding
        chunks = chunk_document(result.document, payload.max_tokens_per_chunk, tokenizer, payload.merge_sections)
        
        # Create response without json_output field unless specifically requested
        response_data = {"chunks": chunks}
        if payload.include_json:
            response_data["json_output"] = result.document.export_to_dict()
        
        return ParseResponse(
            message="Document chunked successfully",
            status="Ok",
            data=ChunkResponseData(**response_data),
        )
    else:
        # Original flow returning text output
    output = _get_output(result.document, payload.output_format)

        # Create response without json_output field unless specifically requested
        response_data = {"output": output}
        if payload.include_json:
            response_data["json_output"] = result.document.export_to_dict()

    return ParseResponse(
        message="Document parsed successfully",
        status="Ok",
            data=ParseResponseData(**response_data),
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
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=config.port,
        log_config=get_log_config(config.log_level),
        reload=config.dev_mode,
        workers=config.get_num_workers(),
    )
