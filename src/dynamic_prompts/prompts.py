import logging
import argparse
import ollama
from config import ROLES, PROMPT_GENERATION_MODEL, DB_CONFIG, TOP_K
from retriever import build_context, retrieve_similar_chunks
from db import DBConnection
logger = logging.getLogger(__name__)

def generate_prompt(context: str, question: str, role: str) -> str:
    if role not in ROLES:
        raise ValueError(f"Invalid role: {role}")

    role_desc = ROLES[role]

    instruction = f"""
                    Context:
                    {context}
                    Task:
                    Using the above context, write a prompt to ask a {role} agent {question}.
                    Critical Instructions:
                    1. You are NOT writing a response as the {role}. 
                    2. You are writing the prompt that a user would use to ASK the {role} agent.
                    3. The prompt should be tailored to the {role}'s focus, which is: {role_desc}.
                    4. Output ONLY the generated prompt. Do not include introductory or concluding text.
                    """.strip()

    try:
        response = ollama.chat(
            model=PROMPT_GENERATION_MODEL,
            messages=[{"role": "user", "content": instruction}]
        )
        return response["message"]["content"].strip()

    except Exception as e:
        logging.exception("Failed to generate prompt")
        raise RuntimeError(f"Ollama request failed: {e}")
    
def generate_prompt_from_rag(question: str, role: str, top_k: int = TOP_K) -> str:
    with DBConnection(**DB_CONFIG) as db:
        chunks = retrieve_similar_chunks(db=db, question=question, top_k=top_k)

    if not chunks:
        raise ValueError("No relevant context found in the database.")

    context = build_context(chunks)
    return generate_prompt(context=context, question=question, role=role)
