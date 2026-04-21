## Overview

The `dynamic_prompts` module facilitates the creation of high-quality, persona-aligned prompts for various healthcare and clinical roles. By combining semantic search over historical document chunks with dynamic templating, it ensures that generated instructions are grounded in available context and tailored to specific professional perspectives.

## Key Components

### 1. `prompts.py` (Core Logic)
The primary entry point for prompt generation.
- **`generate_prompt`**: Creates a prompt for a specified role based on provided context and a specific question.
- **`generate_prompt_from_rag`**: Orchestrates the full RAG pipeline—retrieving relevant context from the database before generating the final prompt.

### 2. `retriever.py` (Semantic Search)
Handles the retrieval layer using vector embeddings.
- **`get_embedding`**: Generates vector representations of queries using `ollama.embeddings`.
- **`retrieve_similar_chunks`**: Performs a vector similarity search (`pgvector`) against the `documents` table in PostgreSQL.
- **`build_context`**: Formats retrieved chunks into a coherent block for injection into LLM templates.

### 3. `db.py` (Persistence Layer)
A PostgreSQL connection wrapper using `psycopg2` with `RealDictCursor` support for easy integration with Python dictionaries.

### 4. `llm_prompts.py` (Evaluation & Bulk Generation)
A utility script for benchmarking and bulk operations.
- Generates prompts across multiple roles (Nurse, Physician Assistant, PhD, etc.) and multiple models (Llama 3, Mistral, Gemma).
- Loads test cases from JSON and exports results to Excel (`.xlsx`) or CSV for analysis.

## Technical Stack

- **Large Language Models**: Powered by [Ollama](https://ollama.ai/) (Llama3, Mistral, Gemma).
- **Database**: PostgreSQL with the `pgvector` extension.
- **Embeddings**: Local embedding models via Ollama.
- **Data Handling**: `pandas` and `openpyxl` for evaluation exports.

## Getting Started

### Prerequisites
1.  **Ollama**: Ensure Ollama is installed and the required models are pulled:
    ```bash
    ollama pull llama3
    ollama pull mistral
    ollama pull mxbai-embed-large  # or your preferred embedding model
    ```
2.  **PostgreSQL**: A database with `pgvector` enabled and a `documents` table schema matching the retrieval query structure.

### Usage Example

```python
from dynamic_prompts.prompts import generate_prompt_from_rag

# Generate a role-specific prompt using the RAG pipeline
question = "How should we manage this patient's post-operative mobility?"
role = "Physical Therapist"

generated_prompt = generate_prompt_from_rag(question, role)
print(generated_prompt)
```

## Running Evaluation

To generate a comparison matrix across models and roles:
```bash
python llm_prompts.py input_cases.json
```
