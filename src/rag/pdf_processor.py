from datetime import datetime
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import detect_scenario_with_groq, build_chunk_metadata
from pypdf import PdfReader
import io
import requests


def download_and_extract_pdf(url, headers):
    """Download and extract text from PDF"""
    try:
        response = requests.get(url, headers=headers)

        if url.endswith(".pdf") and response.status_code == 200:
            reader = PdfReader(io.BytesIO(response.content))

            pages_data = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    pages_data.append({
                        'page_number': page_num + 1,
                        'text': text
                    })

            return pages_data
        else:
            print(f"Failed to download. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return None


def process_pdf(config, groq_client, text_splitter):
    """Process a single PDF and return documents"""

    print(f"\n{'='*80}")
    print(f"Processing: {config['source_title']}")
    print(f"PDF ID: {config['pdf_id']}")
    print(f"{'='*80}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
    }

    # Download and extract PDF
    print("Downloading PDF...")
    pages_data = download_and_extract_pdf(config['url'], headers)

    if not pages_data:
        return []

    print(f"Extracted text from {len(pages_data)} pages")

    # Create documents with page numbers
    docs_with_pages = []
    for page_data in pages_data:
        doc = Document(
            page_content=page_data['text'],
            metadata={'page': page_data['page_number']}
        )
        docs_with_pages.append(doc)

    # Split documents while preserving page numbers
    print("Splitting into chunks...")
    chunked_docs = text_splitter.split_documents(docs_with_pages)
    print(f"Created {len(chunked_docs)} chunks")

    # Process chunks with scenario detection
    documents = []
    scenario_distribution = {}

    print("\nDetecting scenario types...")
    for chunk_index, chunk_doc in enumerate(chunked_docs):
        print(f"  Chunk {chunk_index + 1}/{len(chunked_docs)}... ", end="")

        chunk_text = chunk_doc.page_content
        scenario_type = detect_scenario_with_groq(chunk_text, groq_client)
        print(f"→ {scenario_type}")

        scenario_distribution[scenario_type] = scenario_distribution.get(
            scenario_type, 0) + 1

        page_number = chunk_doc.metadata.get('page', 1)

        # Build metadata using utility function
        metadata = build_chunk_metadata(
            chunk_text=chunk_text,
            scenario_type=scenario_type,
            config=config,
            chunk_index=chunk_index,
            source_type='pdf',
            page_number=page_number
        )

        doc = Document(
            page_content=chunk_text,
            metadata=metadata
        )
        documents.append(doc)

    # Print scenario distribution for this PDF
    print("\n  Scenario Type Distribution:")
    for scenario, count in sorted(scenario_distribution.items()):
        percentage = (count / len(documents)) * 100
        print(f"    {scenario}: {count} chunks ({percentage:.1f}%)")

    return documents
