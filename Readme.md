## BRIDGE: Building Role-specific Interprofessional Dialogue and Guided Education (Version 2.0)

BRIDGE is an AI-powered simulation platform designed to transform interprofessional education (IPE) in healthcare. By leveraging a multi-agent architecture and Retrieval-Augmented Generation (RAG), the system allows students to interact with a panel of virtual healthcare professionals who provide grounded, role-specific perspectives on complex clinical case studies.

## Project Purpose & Utility

In real-world healthcare, effective collaboration between disciplines is critical for patient outcomes. However, early-career professionals often lack exposure to how different roles—such as nurses, physician assistants, and social workers—approach the same medical scenario.

**BRIDGE - version2 addresses this gap by:**
- **Simulating Expert Perspectives**: Providing high-fidelity responses from six distinct healthcare professions.
- **Grounding in Evidence**: Replacing generic AI responses with evidence-based reasoning derived from verified clinical literature (CDC, SAMHSA, NIDA).
- **Scaling IPE Training**: Enabling unlimited practice with realistic, high-stakes scenarios like opioid overdose management and long-term recovery planning.

## Architecture Evolution

### Previous Architecture (Phase 1)
The initial iteration of BRIDGE relied solely on the **parametric knowledge** of Large Language Models (LLMs).
- **Logic**: A simple multi-agent loop using system prompts to define roles.
- **Limitations**:
    - **Hallucinations**: No grounding in real clinical guidelines.
    - **Repetition**: Agents often echoed each other due to shared training data.
    - **Genericness**: Lack of depth in domain-specific reasoning (e.g., specific dosage titration or social service referrals).
    - **Infrastructure**: Lacked a structured data ingestion pipeline and staging environment.

### Current Architecture (Production RAG)
The current version introduces a sophisticated **Retrieve-Then-Generate** architecture that grounds every agent response in a curated knowledge base.

- **Unified RAG Pipeline**: A modular extraction system that transforms PDFs and web resources into segmentable data.
- **Profession-Filtered Retrieval**: Unlike global search, each agent queries a partitioned index filtered by its professional category. This ensures a Nurse retrieves nursing-specific evidence while a Social Worker retrieves community resources.
- **Core Technology Stack**:
    - **LLM**: OpenAI `gpt-5.2` (Reasoning & Synthesis).
    - **Embedding Model**: `nomic-embed-text`
    - **Vector Store**: 
        - For testing: [Pinecone](https://www.pinecone.io/)         
        - For production: Amazon RDS with pgvector extension (Metadata-filtered production storage).
    - **Backend**: FastAPI (Python) orchestrating parallel agent execution.
- **Safety & Referral System**: A decision layer that identifies out-of-scope or high-risk student queries and recommends escalation to real-world experts.

## Project Structure

```text
spring-2026-group1/
├── src/
│   ├── rag/                 # Core RAG Engineering
│   │   ├── pdf_extractor/   # Docling/PDFPlumber ingestors
│   │   ├── web_extractor/   # Trafilatura-based web crawlers
│   │   ├── pdf_chunker/     # Multi-strategy segmentation logic
│   │   └── embedding_retrieval/ # Pinecone & Model evaluation suite
│   └── dynamic_prompts/     # Role-based prompt templates & LLM orchestrators
├── proposals/               # Project vision and technical roadmaps
├── reports/                 # Comprehensive benchmarking and analytics
├── demo/                    
├── figs/                    # Project architecture diagrams
├── presentation/            # Project presentations
├── research_paper/          # Research papers
└── requirements.txt         # Production dependencies
```
