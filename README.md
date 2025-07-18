# Tiny-Tune: LLM Experimentation Repository

A comprehensive repository demonstrating various aspects of working with Large Language Models (LLMs), including prompting, fine-tuning, RAG systems, and agents.

## Overview

This repository contains Jupyter notebooks and code examples exploring different aspects of LLM development and deployment:

### Local LLM Integration with LM Studio
Several notebooks use LM Studio for local LLM inference:
- `04-RAG-using-frameworks.ipynb`: Demonstrates RAG (Retrieval Augmented Generation) using local Mistral model
  - Uses OpenAI-compatible API
  - Integrates with LlamaIndex for document processing
  - Shows both direct API usage and framework integration

### Notebooks
1. `01-instruction-tuning.ipynb`: Basic instruction fine-tuning techniques
2. `02-peft-tuning.ipynb`: Parameter-Efficient Fine-Tuning approaches
3. `03-ultra-simple-RAG.ipynb`: Simple RAG implementation
4. `04-RAG-using-frameworks.ipynb`: Advanced RAG using LlamaIndex and LM Studio
5. `05-tool-calling-langgraph.ipynb`: Tool use and agent frameworks with LangGraph

### Features
- 🤖 Local LLM Integration
- 📚 RAG Systems
- 🔧 Fine-Tuning Examples
- 🛠️ Tool-Calling Agents
- 📊 Framework Comparisons (LlamaIndex vs LangChain)

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/siddBanPsu/tiny-tune.git
cd tiny-tune
```

2. Install uv (if not already installed):
```bash
pip install uv  # or brew install uv
```

3. Create and activate a virtual environment:
```bash
uv venv
# Activate the environment
source .venv/bin/activate  # On Unix/macOS
# or
.\.venv\Scripts\activate   # On Windows
```

4. Install dependencies:
```bash
uv pip install -r requirements.txt  # Faster installation with uv
```

5. For notebooks using LM Studio:
   - Download and install [LM Studio](https://lmstudio.ai/)
   - Load your preferred model (e.g., Mistral)
   - Start the local server (usually runs on port 1234). Check the port number in LM Studio's settings or logs to confirm, as it may vary depending on your configuration. If needed, update the port in your application settings to match the one used by LM Studio.

## Directory Structure
```
tiny-tune/
├── notebooks/              # Jupyter notebooks
│   ├── data/              # Sample data files
│   ├── sft_output/        # Fine-tuning outputs
│   └── storage/           # Vector store data
├── requirements.txt       # Python dependencies
└── README.md             # This file
```
