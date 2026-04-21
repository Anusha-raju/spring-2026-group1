import sys
import json
import ollama
import pandas as pd


CONTEXT = """"""
QUESTION = ""
ROLES = {
    "Nurse": "Patient care, empathy, and practical nursing observations",
    "Physician Assistant": "Medical diagnostics, clinical decision-making, and treatment planning",
    "Medical Social Worker": "Social care, support services, and community resources",
    "Physical Therapist": "Patient rehabilitation, physical therapy techniques, and recovery plans",
    "Public Health Professional": "Population health, disease prevention, health education, and community well-being",
    "Health Administrator": "Organizational management, resource coordination, policy implementation, and interdisciplinary team support"
}


MODELS = ["llama3:latest", "mistral:latest", "gemma:latest"]

def get_instruction(role, role_desc):
    return f"""Context:
{CONTEXT}

Task:
Using the above context, write a prompt to ask a {role} agent {QUESTION}.

Critical Instructions:
1. You are NOT writing a response as the {role}. 
2. You are writing the prompt that a user would use to ASK the {role} agent.
3. The prompt should be tailored to the {role}'s focus, which is: {role_desc}.
4. Output ONLY the generated prompt. Do not include introductory or concluding text."""


def load_cases(json_path: str):
    """
    Accepts either:
      1) a list of objects: [{"context": "...", "question": "..."}, ...]
      2) an object with a key containing that list (common patterns): {"cases": [...]} or {"data": [...]}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        cases = data
    elif isinstance(data, dict):
        # Try common container keys; fall back to error if not found
        for key in ("cases", "data", "items", "inputs"):
            if key in data and isinstance(data[key], list):
                cases = data[key]
                break
        else:
            raise ValueError(
                "JSON must be a list of {context, question} objects, or a dict containing one under keys: "
                "cases/data/items/inputs."
            )
    else:
        raise ValueError("Unsupported JSON format. Expected a list or dict.")

    # Validate and normalize
    normalized = []
    for i, item in enumerate(cases):
        if not isinstance(item, dict):
            raise ValueError(f"Case #{i} is not an object/dict.")
        if "context" not in item or "question" not in item:
            raise ValueError(f"Case #{i} missing required keys. Found keys: {list(item.keys())}")
        normalized.append({
            "case_id": i,
            "context": str(item["context"]),
            "question": str(item["question"]),
        })

    return normalized


def main():
    # Usage: python script.py input.json
    if len(sys.argv) < 2:
        print("Usage: python script.py <input.json>")
        sys.exit(1)

    input_json = sys.argv[1]
    cases = load_cases(input_json)

    results = []

    print("Starting generation of prompts using local open-source LLMs via Ollama...\n")
    print(f"Loaded {len(cases)} case(s) from: {input_json}\n")

    global CONTEXT, QUESTION

    for case in cases:
        CONTEXT = case["context"]
        QUESTION = case["question"]

        print(f"\n==================== Case {case['case_id']} ====================")

        for model in MODELS:
            print(f"\n========== Loading & Generating with Model: {model} ==========")

            for role, desc in ROLES.items():
                print(f"  -> Generating for Role: {role}...")
                instruction = get_instruction(role, desc)

                try:
                    response = ollama.chat(model=model, messages=[
                        {'role': 'user', 'content': instruction}
                    ])
                    generated_prompt = response['message']['content'].strip()
                except Exception as e:
                    print(f"    [Error] Failed to get response from {model}.")
                    print(f"    Make sure Ollama is running and you have pulled the model (`ollama pull {model}`). Details: {e}")
                    generated_prompt = f"ERROR: {e}"

                results.append({
                    "case_id": case["case_id"],
                    "context": CONTEXT,
                    "question": QUESTION,
                    "role": role,
                    "opensourcellm": model,
                    "generated prompt": generated_prompt
                })

    # Save to Excel
    output_filename = "generated_prompts.xlsx"
    print(f"\nSaving results to {output_filename}...")

    try:
        df = pd.DataFrame(results)
        df.to_excel(output_filename, index=False)
        print(f"Success! Data written to {output_filename}")
    except Exception as e:
        print(f"Error saving to Excel (ensure 'openpyxl' is installed): {e}")
        csv_filename = "generated_prompts.csv"
        print(f"Attempting to save as CSV instead: {csv_filename}")
        df.to_csv(csv_filename, index=False)
        print("Saved to CSV.")

if __name__ == "__main__":
    main()