"""
Utility functions for scenario detection and analysis
"""
from datetime import datetime

OPIOID_SCENARIO_TYPES = [
    "opioid_overdose_response",
    "opioid_initiation_and_prescribing",
    "opioid_tapering_and_withdrawal_management",
    "opioid_use_disorder_screening_and_referral",
    "opioid_chronic_pain_management_and_monitoring",
    "general_opioid_information"
]


def detect_scenario_with_groq(chunk_text, client):
    """
    Using Groq API to detect scenario type

    Args:
        chunk_text (str): The text chunk to classify
        client: Groq client instance

    Returns:
        str: One of the OPIOID_SCENARIO_TYPES
    """

    prompt = f"""Analyze this text chunk and classify it into ONE of these opioid-related scenario types:

1. opioid_overdose_response - Content about recognizing and responding to overdoses, naloxone administration
2. opioid_initiation_and_prescribing - Content about when and how to prescribe opioids, initial assessment
3. opioid_tapering_and_withdrawal_management - Content about reducing opioid doses, managing withdrawal symptoms
4. opioid_use_disorder_screening_and_referral - Content about identifying OUD, screening tools, treatment referrals
5. opioid_chronic_pain_management_and_monitoring - Content about long-term pain management, monitoring patients on opioids
6. general_opioid_information - General information that doesn't fit other categories

Text chunk:
{chunk_text}

Respond with ONLY the scenario type identifier (e.g., "opioid_overdose_response"). No explanation or additional text."""

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        detected_scenario = completion.choices[0].message.content.strip(
        ).lower()

        # Validate the response
        if detected_scenario in OPIOID_SCENARIO_TYPES:
            return detected_scenario
        else:
            # Try to find a partial match
            for scenario_type in OPIOID_SCENARIO_TYPES:
                if scenario_type in detected_scenario:
                    return scenario_type

            # Default fallback
            print(
                f"    Warning: Unexpected scenario '{detected_scenario}', defaulting to general_opioid_information")
            return "general_opioid_information"

    except Exception as e:
        print(f"    Error detecting scenario: {e}")
        return "general_opioid_information"


def build_chunk_metadata(
    chunk_text,
    scenario_type,
    config,
    chunk_index,
    source_type='web',
    **extra_fields
):
    """
    Build standardized metadata for a document chunk

    Args:
        chunk_text (str): The actual text content
        scenario_type (str): Detected scenario type
        config (dict): Configuration dictionary with source info
        chunk_index (int): Index of this chunk in the source
        source_type (str): 'web' or 'pdf'
        **extra_fields: Additional metadata fields (page_number, heading_path, token_count, etc.)

    Returns:
        dict: Standardized metadata dictionary
    """
    # Get source ID (handle both pdf_id and source_id)
    source_id = config.get('pdf_id') or config.get('source_id', 'unknown')

    # Build chunk ID based on source type
    if source_type == 'pdf':
        page_number = extra_fields.get('page_number', 1)
        chunk_id = f"{source_id}_p{page_number:02d}_c{chunk_index:02d}"
    else:
        chunk_id = f"{source_id}_c{chunk_index:03d}"

    # Base metadata
    metadata = {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "role": config.get('role', 'general'),
        "scenario_type": scenario_type,
        "source_title": config.get('source_title', 'Unknown'),
        "source_link": config.get('url', ''),
        "chunk_index": chunk_index,
        "created_at": datetime.now().strftime("%Y-%m-%d"),
        "version": "v1",
        "word_count": len(chunk_text.split()),
        "char_count": len(chunk_text),
    }

    # Add source-type specific fields
    if source_type == 'pdf':
        metadata["pdf_id"] = source_id
        metadata["page_number"] = extra_fields.get('page_number', 1)
    else:
        metadata["heading_path"] = extra_fields.get('heading_path', 'Document')
        metadata["token_count"] = extra_fields.get('token_count', 0)
    for key, value in extra_fields.items():
        if key not in metadata:
            metadata[key] = value

    return metadata


def overall_scenario_distribution(all_documents):
    """
    Calculate and print overall scenario type distribution

    Args:
        all_documents (list): List of Document objects with metadata
    """
    overall_scenario_dist = {}

    for doc in all_documents:
        scenario = doc.metadata.get('scenario_type', 'unknown')
        overall_scenario_dist[scenario] = overall_scenario_dist.get(
            scenario, 0) + 1

    print("\nOverall Scenario Type Distribution:")
    print(f"{'Scenario Type':<50} {'Count':<10} {'Percentage'}")
    print("-" * 70)

    for scenario, count in sorted(overall_scenario_dist.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(all_documents)) * 100
        print(f"{scenario:<50} {count:<10} {percentage:>6.1f}%")

    print("-" * 70)
    print(f"{'Total':<50} {len(all_documents):<10} {100.0:>6.1f}%")


def get_scenario_summary_by_source(all_documents):
    """
    Get scenario distribution grouped by source

    Args:
        all_documents (list): List of Document objects with metadata

    Returns:
        dict: Nested dict with source -> scenario -> count
    """
    source_scenarios = {}

    for doc in all_documents:
        source_id = doc.metadata.get(
            'source_id') or doc.metadata.get('pdf_id', 'unknown')
        scenario = doc.metadata.get('scenario_type', 'unknown')

        if source_id not in source_scenarios:
            source_scenarios[source_id] = {}

        source_scenarios[source_id][scenario] = source_scenarios[source_id].get(
            scenario, 0) + 1

    return source_scenarios


def print_scenario_summary_by_source(all_documents):
    """
    Print scenario distribution grouped by source

    Args:
        all_documents (list): List of Document objects with metadata
    """
    source_scenarios = get_scenario_summary_by_source(all_documents)

    print("\n" + "="*80)
    print("SCENARIO DISTRIBUTION BY SOURCE")
    print("="*80)

    for source_id, scenarios in sorted(source_scenarios.items()):
        total_chunks = sum(scenarios.values())
        print(f"\n{source_id} ({total_chunks} chunks):")
        print("-" * 60)

        for scenario, count in sorted(scenarios.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_chunks) * 100
            print(f"  {scenario:<45} {count:>4} ({percentage:>5.1f}%)")
