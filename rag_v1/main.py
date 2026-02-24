import os
import requests
import trafilatura
from web_processor import extract_and_chunk
from pdf_parsing import HybridPDFParser
from utils import detect_scenario_with_groq, overall_scenario_distribution, build_chunk_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from groq import Groq
from rag_v1.embedding_processor import EmbeddingProcessor

configs = [
    {
        'url': "https://www.chcs.org/resource/a-federally-qualified-health-center-and-certified-community-behavioral-health-clinic-partnership-in-rural-missouri/",
        'source_id': 'web001',
        'role': 'nurse',
        'source_title': "FQHC and CCBHC Partnership in Rural Missouri"
    },
    {
        'url': "https://www.asam.org/docs/default-source/quality-science/a-drug-court-team-member%27s-guide-to-medication-in-addiction-treatment.pdf",
        'pdf_path': "src/pdf/pdf1.pdf",
        'pdf_id': 'pdf001',
        'role': 'social_worker',
        'source_title': "A Drug Court Team Member's Guide to Medication in Addiction Treatment",
    },
]

# Initialize Groq client
GROQ_API_KEY = "groq_key"
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize text splitter for PDFs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=256,
    length_function=len
)

# Web scraping parameters
MAX_TOKENS = 400
MIN_TOKENS = 100
OVERLAP_TOKENS = 40
MERGE_OVERFLOW = 20
all_documents = []
chunk_output_path = "outputs/chunks_output.jsonl"

for config in configs:
    print(f"\n{'='*80}")
    print(f"Processing: {config['source_title']}")
    print(f"{'='*80}")

    url = config['url']

    if url.endswith('.pdf') or config.get("pdf_path"):
        # Process PDF using hybrid parser
        pdf_path = config.get("pdf_path")
        if not pdf_path:
            print("⚠️  pdf_path not provided for PDF config; skipping.")
            continue
        parser = HybridPDFParser()
        result = parser.parse(pdf_path)
        output_dir = "hybrid_output"
        os.makedirs(output_dir, exist_ok=True)
        md_path = os.path.join(output_dir, f"{config.get('pdf_id', 'pdf')}.md")
        parser.save_markdown(result, md_path)

    else:
        # Process web page
        print("Fetching web page...")
        html = trafilatura.fetch_url(url)
        if not html:
            html = requests.get(url, timeout=15).text

        # Extract and chunk the content
        chunks = extract_and_chunk(
            html,
            url=url,
            max_tokens=MAX_TOKENS,
            min_tokens=MIN_TOKENS,
            overlap_tokens=OVERLAP_TOKENS,
            merge_overflow=MERGE_OVERFLOW
        )

        print(f"Created {len(chunks)} chunks")

        # Process chunks with scenario detection
        web_documents = []
        scenario_distribution = {}

        # print("\nDetecting scenario types...")
        for chunk_index, chunk in enumerate(chunks):
            # print(f"  Chunk {chunk_index + 1}/{len(chunks)}... ", end="")

            chunk_text = chunk['text']
            # scenario_type = detect_scenario_with_groq(chunk_text, groq_client)
            scenario_type = "general_opioid_information"
            # print(f"→ {scenario_type}")

            scenario_distribution[scenario_type] = scenario_distribution.get(
                scenario_type, 0) + 1

            # Build metadata using utility function
            metadata = build_chunk_metadata(
                chunk_text=chunk_text,
                scenario_type=scenario_type,
                config=config,
                chunk_index=chunk_index,
                source_type='web',
                heading_path=chunk.get('heading_path', 'Document'),
                token_count=chunk.get('tokens', 0)
            )
            doc = Document(
                page_content=chunk_text,
                metadata=metadata
            )
            web_documents.append(doc)
        all_documents.extend(web_documents)

        # Print scenario distribution
        print("\n  Scenario Type Distribution:")
        for scenario, count in sorted(scenario_distribution.items()):
            percentage = (count / len(web_documents)) * 100
            print(f"    {scenario}: {count} chunks ({percentage:.1f}%)")

        # Write chunk output to a file (JSONL)
        os.makedirs(os.path.dirname(chunk_output_path), exist_ok=True)
        with open(chunk_output_path, "a", encoding="utf-8") as f:
            for c in chunks:
                record = {
                    "source_title": config.get("source_title", "Unknown"),
                    "source_url": url,
                    "heading_path": c.get("heading_path", "Document"),
                    "tokens": c.get("tokens", 0),
                    "text": c.get("text", ""),
                }
                f.write(f"{record}\n")

# Final Summary
print(f"\n{'='*80}")
print(f"FINAL SUMMARY")
print(f"{'='*80}")

# Count PDFs vs Web sources
pdf_count = sum(1 for config in configs if config['url'].endswith('.pdf'))
web_count = len(configs) - pdf_count

print(f"Total sources processed: {len(configs)}")
print(f"  - PDFs: {pdf_count}")
print(f"  - Web pages: {web_count}")
print(f"Total chunks created: {len(all_documents)}")

# Overall scenario distribution
overall_scenario_distribution(all_documents)

# Generate embeddings and store in vector database
print(f"\n{'='*80}")
print(f"GENERATING EMBEDDINGS AND STORING IN VECTOR DATABASE")
print(f"{'='*80}")

try:
    embedding_processor = EmbeddingProcessor(
        embedding_provider="sentence_transformers",
        vector_db="pgvector",
        collection_name="opioid_documents",
        pg_dsn=os.getenv("PGVECTOR_DSN"),
    )

    # Store all documents with embeddings
    embedding_processor.store_documents(
        documents=all_documents,
        batch_size=100
    )

    # Get and display stats
    stats = embedding_processor.get_collection_stats()
    print(f"\n✓ Vector Database Stats:")
    print(f"  Total documents stored: {stats['total_documents']}")
    print(f"  Embedding dimension: {stats['embedding_dimension']}")
    print(f"  Collection name: {stats['collection_name']}")

    # Example search to verify it works
    print(f"\n{'='*80}")
    print(f"EXAMPLE SEARCH (Testing Vector Database)")
    print(f"{'='*80}")

    query = "How to respond to an opioid overdose?"
    print(f"Query: '{query}'")

    results = embedding_processor.search(
        query=query,
        n_results=3,
        filters={"role": "nurse", "scenario_type": "opioid_overdose_response"},
    )

    print(f"\nFound {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Chunk ID: {result['id']}")
        print(f"   Scenario: {result['metadata'].get('scenario_type', 'N/A')}")
        print(f"   Source: {result['metadata'].get('source_title', 'N/A')}")
        print(f"   Distance: {result.get('distance', 'N/A'):.4f}")
        print(f"   Preview: {result['text'][:200]}...")
        print()

except ImportError:
    print("\n Embedding processor not available. Install dependencies:")
except Exception as e:
    print(f"\n Error during embedding generation: {e}")

print(f"\n{'='*80}")
print("Processing complete!")
print(f"{'='*80}")
