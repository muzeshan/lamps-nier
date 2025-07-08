import os
import tarfile
import requests
from typing import List
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from crewai import Agent
import torch

# === STEP 1: Load Meta-LLaMA 3 (for reasoning agents) ===
llama_model_id = "meta-llama/Llama3-8B-Instruct"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_id, device_map="auto", torch_dtype=torch.float16
)
llama_pipeline = pipeline(
    "text-generation", model=llama_model, tokenizer=llama_tokenizer, max_new_tokens=512
)

class LlamaWrapper:
    def run(self, prompt: str) -> str:
        full_prompt = f"""
You are an expert assistant helping with PyPI package analysis.

Task:
{prompt}

Respond concisely and clearly.
"""
        return llama_pipeline(full_prompt)[0]['generated_text'].strip()


# === STEP 2: Load fine-tuned CodeBERT (for classification agent) ===
codebert_model_path = "models/codebert-malware-detector"  # <-- update with your path
codebert_model = AutoModelForSequenceClassification.from_pretrained(codebert_model_path)
codebert_tokenizer = AutoTokenizer.from_pretrained(codebert_model_path)
codebert_pipeline = pipeline("text-classification", model=codebert_model, tokenizer=codebert_tokenizer)

class CodeBERTClassifier:
    def run(self, code: str) -> dict:
        prompt = f"""
You are a security expert. Classify the following Python code as either malicious or benign.

Malicious code may include obfuscation, data exfiltration, network exploitation, or hidden behavior.

Code:
{code[:512]}
"""
        result = codebert_pipeline(prompt)[0]
        return {
            "label": result["label"],
            "score": result["score"]
        }


# === STEP 3: Define CrewAI agents ===
llm = LlamaWrapper()
codebert_llm = CodeBERTClassifier()

repo_fetcher = Agent(
    role="Package Harvester",
    goal="Retrieve the PyPI source archive (.tar.gz) URL.",
    backstory="Knows PyPI API and metadata structure.",
    llm=llm,
    verbose=True
)

extractor = Agent(
    role="Code File Extractor",
    goal="Extract .py files from the package archive.",
    backstory="Handles complex archive structures.",
    llm=llm,
    verbose=True
)

classifier = Agent(
    role="Security Code Classifier",
    goal="Classify each Python file as malicious or benign using CodeBERT.",
    backstory="Trained to detect malicious logic using fine-tuned CodeBERT.",
    llm=codebert_llm,
    verbose=True
)

verdict_agent = Agent(
    role="Package Verdict Synthesizer",
    goal="Based on classification results, declare if the package is malicious and justify.",
    backstory="Synthesizes threat evidence for risk judgement.",
    llm=llm,
    verbose=True
)


# === STEP 4: Core utility functions ===
def fetch_pypi_url(package_name: str) -> str:
    print(f"\U0001F4E6 Fetching metadata for {package_name}")
    resp = requests.get(f"https://pypi.org/pypi/{package_name}/json").json()
    for entry in resp['urls']:
        if entry['packagetype'] == 'sdist':
            return entry['url']
    raise Exception("No source distribution found.")

def download_file(url: str, download_dir='downloads') -> str:
    os.makedirs(download_dir, exist_ok=True)
    path = os.path.join(download_dir, url.split("/")[-1])
    print(f"⬇️  Downloading: {url}")
    with requests.get(url, stream=True) as r:
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return path

def extract_py_files(archive_path: str, extract_dir='extracted') -> List[str]:
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    py_files = []
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return py_files

def classify_file(filepath: str) -> dict:
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        code = f.read()
    result = codebert_llm.run(code)
    return {
        "file": filepath,
        "label": result['label'],
        "score": result['score']
    }

def final_verdict(results: List[dict]) -> tuple:
    malicious_files = [r for r in results if r['label'].lower() == 'malicious']
    if malicious_files:
        explanation = "\n".join([f"{r['file']} → malicious ({r['score']:.2f})" for r in malicious_files])
        return "Malicious", explanation
    else:
        return "Benign", "All files classified as benign."


# === STEP 5: Orchestrate full workflow ===
def run_pipeline(package: str):
    url = fetch_pypi_url(package)
    archive_path = download_file(url)
    py_files = extract_py_files(archive_path)
    print(f"\U0001F9E0 Classifying {len(py_files)} files...\n")
    results = [classify_file(f) for f in py_files]
    verdict, explanation = final_verdict(results)
    print(f"\n\U0001F4E6 Final Verdict: {verdict}")
    print("\U0001F4DD Reasoning:\n", explanation)


# === Entry point ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid CrewAI + CodeBERT PyPI Malware Classifier")
    parser.add_argument("--package", type=str, required=True, help="PyPI package name (e.g., requests)")
    args = parser.parse_args()
    run_pipeline(args.package)