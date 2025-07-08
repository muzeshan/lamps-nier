# Replication Package: LAMPS

This package includes a multi-agent pipeline using CrewAI with:
- Meta-LLaMA 3 (Instruct) for reasoning agents
- Fine-tuned CodeBERT classifier for Python file classification (malicious/benign)

## Structure

- `hybrid_pypi_classifier.py`: Main executable pipeline (CrewAI-MAS)
- `models/codebert-malware-detector`: fine-tuned CodeBERT model
- `downloads/`: PyPI packages
- `Dataset/`: dataset used for fine-tuning classifier
- `extracted/`: Python files extracted
- `prompts.md`: Prompts structure used in the pipeline


- `README.md`: This file

## Usage

Install dependencies:
```bash
pip install crewai transformers accelerate einops requests tqdm
```

Run the classifier:
```bash
python hybrid_pypi_classifier.py --package <pypi-package-name>
```

Example:
```bash
python hybrid_pypi_classifier.py --package requests
```
