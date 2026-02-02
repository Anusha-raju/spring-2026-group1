from groq import Groq
import json
from datetime import datetime
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import io
import requests

# Define URLs and their metadata
pdf_configs = [
    {
        'url': "https://www.asam.org/docs/default-source/quality-science/a-drug-court-team-member%27s-guide-to-medication-in-addiction-treatment.pdf",
        'pdf_id': 'pdf001',
        'role': 'social_worker',
        'source_title': "A Drug Court Team Member's Guide to Medication in Addiction Treatment",
    },
    {
        'url': "https://www.oregonpainguidance.org/wp-content/uploads/2020/06/13.-CDC-Why-Guidelines-for-Primary-Care-Providers.pdf",
        'pdf_id': 'pdf002',
        'role': 'nurse',
        'source_title': "CDC Guidelines for Primary Care Providers",
    },
]

# Define the valid scenario types
OPIOID_SCENARIO_TYPES = [
    "opioid_overdose_response",
    "opioid_initiation_and_prescribing",
    "opioid_tapering_and_withdrawal_management",
    "opioid_use_disorder_screening_and_referral",
    "opioid_chronic_pain_management_and_monitoring",
    "general_opioid_information"
]


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


def detect_scenario_with_groq(chunk_text, client):
    """Using Groq API to detect scenario type"""

    prompt = f"""Analyze this text chunk and classify it into ONE of these opioid-related scenario types:

1. opioid_overdose_response
2. opioid_initiation_and_prescribing
3. opioid_tapering_and_withdrawal_management
4. opioid_use_disorder_screening_and_referral
5. opioid_chronic_pain_management_and_monitoring

Text chunk:
{chunk_text}

Respond with ONLY the scenario type identifier. No explanation."""

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        detected_scenario = completion.choices[0].message.content.strip(
        ).lower()

        if detected_scenario in OPIOID_SCENARIO_TYPES:
            return detected_scenario
        else:
            return "general_opioid_information"
    except Exception as e:
        print(f"    Error detecting scenario: {e}")
        return "general_opioid_information"


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
        chunk_id = f"{config['pdf_id']}_p{page_number:02d}_c{chunk_index:02d}"

        metadata = {
            "chunk_id": chunk_id,
            "pdf_id": config['pdf_id'],
            "role": config['role'],
            "scenario_type": scenario_type,
            "page_number": page_number,
            "source_title": config['source_title'],
            "source_link": config['url'],
            "chunk_index": chunk_index,
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "version": "v1",
            "word_count": len(chunk_text.split()),
            "char_count": len(chunk_text),
        }

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


# Main processing
GROQ_API_KEY = "api_key"
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=256,
    length_function=len
)

# Process all PDFs
all_documents = []

for config in pdf_configs:
    pdf_documents = process_pdf(config, groq_client, text_splitter)
    all_documents.extend(pdf_documents)
    print(f"\n✓ Total documents so far: {len(all_documents)}\n")

print(f"\n{'='*80}")
print(f"FINAL SUMMARY")
print(f"{'='*80}")
print(f"Total PDFs processed: {len(pdf_configs)}")
print(f"Total chunks created: {len(all_documents)}")

# Overall scenario distribution
overall_scenario_dist = {}
for doc in all_documents:
    scenario = doc.metadata.get('scenario_type')
    overall_scenario_dist[scenario] = overall_scenario_dist.get(
        scenario, 0) + 1

print("\nOverall Scenario Type Distribution:")
for scenario, count in sorted(overall_scenario_dist.items()):
    percentage = (count / len(all_documents)) * 100
    print(f"  {scenario}: {count} chunks ({percentage:.1f}%)")
