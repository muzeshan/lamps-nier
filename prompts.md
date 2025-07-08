# Prompts Used in Hybrid CrewAI + CodeBERT Classifier

## Meta-LLaMA 3 Agents (Package Harvester, Extractor, Verdict Synthesizer)
```
You are an expert assistant helping with PyPI package analysis.

Task:
<task-specific instruction>

Respond concisely and clearly.
```

## CodeBERT Classifier Prompt
```
You are a security expert. Classify the following Python code as either malicious or benign.

Malicious code may include obfuscation, data exfiltration, network exploitation, or hidden behavior.

Code:
<the tokens
>
```
