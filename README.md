# Docling inference server

This project provides a FastAPI wrapper around the
[docling](https://github.com/DS4SD/docling) document parser to make it easier to
use in distributed production environments. This implementation specifically focuses on PDF processing.

## Running

The easiest way to run this project is using docker. There are two image families,
one for cuda machines and one for cpu:

- Cuda: ghcr.io/aidotse/docling-inference:rev
- CPU: ghcr.io/aidotse/docling-inference:cpu-rev

```bash
# Create volumes to not have to download models every time
docker volume create hf_cache
docker volume create ocr_cache

# Run the container
docker run -d \
  --gpus all \
  -p 8080:8080 \
  -e NUM_WORKERS=8 \
  -v hf_cache:/root/.cache/huggingface \
  -v ocr_cache:/root/.EasyOCR \
  ghcr.io/aidotse/docling-inference:latest
```

### Docker compose

```yaml
services:
  docling-inference:
    image: ghcr.io/aidotse/docling-inference:latest
    ports:
      - 8080:8080
    environment:
      - NUM_WORKERS=8
    volumes:
      - hf_cache:/root/.cache/huggingface
      - ocr_cache:/root/.EasyOCR
    restart: always
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  hf_cache:
  ocr_cache:
```

### Local python

Dependencies are handled with [uv](https://docs.astral.sh/uv/) in this
project. Follow their installation instructions if you do not have it.

```bash
# Create a virtual environment
uv venv

# Install the dependencies
uv sync --extra cpu
# OR if you have cuda devices
uv sync --extra cu121

# Install optional PDF optimization
uv pip install pikepdf

# Activate the shell
source .venv/bin/activate

# Start the server
python src/main.py
```

## Using the API

When the server is started you can find the interactive API documentation at the `/docs`
endpoint. If you're running locally with the example command, this will be
`http://localhost:8080/docs`.

You can parse PDF files with the `/parse/file` endpoint:

```sh
curl -X 'POST' \
  'http://localhost:8080/parse/file' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@your-document.pdf;type=application/pdf' \
  -F 'data={"include_json":false,"output_format":"markdown"}'
```

Note: This API only accepts PDF files. Other file formats will result in a 415 Unsupported Media Type error.

### API Options Reference

The `/parse/file` endpoint accepts the following options in the `data` JSON parameter:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_json` | boolean | `false` | Include the full JSON representation of the document in the response |
| `output_format` | string | `"markdown"` | Output format when not using chunking. Options: `"markdown"`, `"text"`, `"html"` |
| `chunk_document` | boolean | `false` | Process the document with Hybrid Chunker and return chunks |
| `max_tokens_per_chunk` | integer | `512` | Maximum tokens per chunk when chunking is enabled |
| `optimize_pdf` | boolean | `true` | Apply PDF optimization before parsing (improves text extraction) |
| `merge_sections` | boolean | `true` | Merge chunks with the same section title across page boundaries |

### Document Chunking

You can request the document to be processed with the Hybrid Chunker, which will split the document into semantic chunks. This is useful for AI applications that have context length limitations:

```sh
curl -X 'POST' \
  'http://localhost:8080/parse/file' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@file-path.pdf;type=application/pdf' \
  -F 'data={"include_json":false,"chunk_document":true,"max_tokens_per_chunk":512}'
```

The response will contain an array of chunks with their text and associated metadata.

### PDF Optimization

The API now performs automatic PDF optimization to improve text extraction quality. This process:
- Removes watermarks and background elements that may interfere with text extraction
- Normalizes font encoding for better character recognition
- Repairs structural issues in the PDF when possible
- Linearizes the document for improved parsing

You can disable this optimization if needed:

```sh
curl -X 'POST' \
  'http://localhost:8080/parse/file' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@file-path.pdf;type=application/pdf' \
  -F 'data={"chunk_document":true,"max_tokens_per_chunk":512,"optimize_pdf":false}'
```

### Section Continuity Across Pages

The API now prioritizes semantic continuity over page boundaries with intelligent section merging:
- Chunks with the same section title are merged across page boundaries
- Merged chunks respect the maximum token limit
- Preserves document flow for better context in RAG applications

This feature ensures that content from the same logical section stays together, even when it spans multiple pages:

```sh
curl -X 'POST' \
  'http://localhost:8080/parse/file' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@file-path.pdf;type=application/pdf' \
  -F 'data={"chunk_document":true,"max_tokens_per_chunk":512,"merge_sections":true}'
```

You can disable this feature if you need to preserve the original page-based chunking:

```sh
curl -X 'POST' \
  'http://localhost:8080/parse/file' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@file-path.pdf;type=application/pdf' \
  -F 'data={"chunk_document":true,"max_tokens_per_chunk":512,"merge_sections":false}'
```

### Advanced Chunking for Embedding

The API now supports optimized chunks for embedding with the Nomic tokenizer. Each chunk includes contextualization (preserving document hierarchy) and rich metadata:

```sh
curl -X 'POST' \
  'http://localhost:8080/parse/file' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@file-path.pdf;type=application/pdf' \
  -F 'data={"include_json":false,"chunk_document":true,"max_tokens_per_chunk":8192}'
```

The returned chunks are specifically designed for embedding in downstream applications:
- Each chunk includes text with proper contextualization (headings and document structure preserved)
- All chunks include the "search_document: " prefix required by Nomic embedding models
- Rich metadata about the document section, including page numbers when available
- Optimal chunk sizes that respect the token limits of embedding models
- Preservation of semantic coherence with merge_peers=True

## Component Defaults & Customization

### Tokenizer

**Default:** 
- For chunking decisions: `sentence-transformers/all-MiniLM-L6-v2` (via HybridChunker)
- For embedding preparation: `nomic-ai/nomic-embed-text-v2-moe` (when available)

**How it works:**
- The server attempts to load the Nomic tokenizer at startup
- If successful, it uses this tokenizer for chunk sizing and formatting
- If unavailable, falls back to the HybridChunker's default tokenizer

**Logs indicate successful initialization:**
```
2025-05-11 00:03:53,423 - src.main - INFO - Nomic tokenizer initialized successfully
```

### Chunk Output Structure

Each chunk in the response includes:

- `section_title`: The heading associated with this chunk (if any)
- `text`: The chunk content with "search_document: " prefix
- `chunk_index`: Sequential position within the document (0-indexed)
- `metadata`: Object containing:
  - `content_type`: Type of content ("text", "table", "heading", "list")
  - `heading_path`: Full breadcrumb path of nested headings
  - `pages`: Page numbers where this chunk appears
  - Additional metadata when applicable (table data, captions, etc.)

Example response chunk:
```json
{
  "section_title": "3.2. Evaluation Criteria",
  "text": "search_document: 3.2. Evaluation Criteria\nProposals will be evaluated based on...",
  "chunk_index": 12,
  "metadata": {
    "content_type": "text",
    "heading_path": "3. Proposal Requirements > 3.2. Evaluation Criteria",
    "pages": [2]
  }
}
```

### OCR Support

**Default languages:** "es,en,fr,de,sv" (Spanish, English, French, German, Swedish)

EasyOCR is used for optical character recognition on image-based PDF content. The list of languages can be customized via the `OCR_LANGUAGES` environment variable.

## Performance Considerations

### Processing Times

Typical processing times based on logs:
- Small PDF (few pages): 3-5 seconds
- Medium PDF (10-30 pages): 10-30 seconds
- Large PDF (30+ pages): 30+ seconds

### Memory Usage

- The server uses MPS (Metal Performance Shaders) acceleration on macOS where available
- For large documents, increase available memory to the server
- Batch processing of very large documents may require queue management

## Troubleshooting

Common errors and their solutions:

1. **Tokenizer sequence length exceeded:**
   ```
   Token indices sequence length is longer than the specified maximum sequence length for this model (514 > 512)
   ```
   Solution: Increase `max_tokens_per_chunk` value

2. **PDF optimization failure:**
   ```
   WARNING - PDF optimization requested but pikepdf is not available
   ```
   Solution: Install pikepdf (`uv pip install pikepdf`)

## Building

Build the project docker image with one of the following commands

- Cuda: `docker build -t ghcr.io/aidotse/docling-inference:dev .`
- CPU: `docker build -f Dockerfile.cpu -t ghcr.io/aidotse/docling-inference:dev .`

## Configuration

Configuration is handled through environment variables. Here is a list of the
available configuration variables. They are defined in `src/config.py`

- `NUM_WORKERS`: The number of processes to run.
- `LOG_LEVEL`: The lowest level of logs to display. One of DEBUG, INFO, WARNING,
  CRITICAL, ERROR.
- `DEV_MODE`: Sets automatic reload of the service. Useful during development
- `PORT`: The port to run the server on.
- `AUTH_TOKEN`: Token to use for authentication. Token is expected in the
  `Authorization: Bearer: <token>` format in the request header. The service is
  unprotected if this option is omitted.
- `OCR_LANGUAGES`: List of language codes to use for optical character optimization.
  Default is `"es,en,fr,de,sv"`. See https://www.jaided.ai/easyocr/ for the list
  of all available languages.
- `DO_CODE_ENRICHMENT`: Use a code enrichment model in the pipeline. Processes images of code to code.
- `DO_FORMULA_ENRICHMENT`: Use a formula enrichment model in the pipeline. Converts formulas to LaTeX.
- `DO_PICTURE_CLASSIFICATION`: Use a picture classification model in the pipelinese. Classifies the type of image into a category.
- `DO_PICTURE_DESCRIPTION`: Use a picture description model in the pipeline. Uses a small multimodal model to describe images.
