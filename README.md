# Radiology Intelligence Assistant

An AI-powered radiology report search and X-ray analysis tool that runs entirely on-premise — no cloud, no external API calls, fully GDPR compliant.

Built as a proof-of-concept for intelligent search over hospital PACS report archives.

---

## What it does

**Tab 1 — Report Search**

Semantic search over 3,337 real radiology reports from the Indiana University chest X-ray dataset. Type a clinical term or finding and the system retrieves the most relevant historical reports using vector similarity search (FAISS + sentence-transformers). A local LLM (llama3.2:3b via Ollama) then summarises the top results into a clinical insight.

**Tab 2 — X-ray Analysis**

Upload a chest X-ray image. A DenseNet121 vision model classifies the likelihood of 14 chest conditions (cardiomegaly, effusion, pneumonia, atelectasis, etc.). The system then automatically searches the report archive for historical cases matching those findings and generates an AI summary.

---

## Why it matters for hospitals

Most AI tools in medical imaging call external cloud APIs — which makes them illegal under GDPR for German hospitals. This system runs 100% locally:

- Embedding model: `all-MiniLM-L6-v2` (cached locally)
- Vector index: FAISS (in-memory, on-device)
- Vision model: DenseNet121 (local inference via PyTorch)
- LLM: llama3.2:3b via Ollama (local)

No patient data, no report text, no images leave the machine.

---

## Tech stack

- Python 3.11
- Streamlit (UI)
- FAISS (vector search)
- sentence-transformers (embeddings)
- PyTorch + torchvision DenseNet121 (X-ray classification)
- Ollama + llama3.2:3b (LLM summary)
- LangChain Core (prompt management)
- Dataset: [Indiana University Chest X-ray Collection](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)

---

## Setup

**Requirements:** Python 3.11, [Ollama](https://ollama.com) installed and running

```bash
# Install dependencies
pip install streamlit faiss-cpu sentence-transformers torch torchvision langchain-core langchain-ollama

# Pull the LLM
ollama pull llama3.2:3b

# Download the dataset (Kaggle account required)
python -m kaggle datasets download -d raddar/chest-xrays-indiana-university --file indiana_reports.csv
unzip indiana_reports.csv.zip

# Run
python -m streamlit run app.py
```

The FAISS index is built on first run and cached as `reports.index`.

---

## Architecture

```
User query / X-ray image
        │
        ▼
┌─────────────────────┐     ┌──────────────────────┐
│  sentence-transformer│     │   DenseNet121         │
│  (text embedding)    │     │   (image classifier)  │
└────────┬────────────┘     └──────────┬───────────┘
         │                             │
         ▼                             ▼
┌─────────────────────────────────────────────────┐
│              FAISS Vector Index                  │
│         (3,337 radiology reports)                │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
           Top-k similar reports
                     │
                     ▼
         ┌───────────────────────┐
         │   llama3.2:3b (Ollama) │
         │   Clinical Summary     │
         └───────────────────────┘
```

---

## Dataset

Indiana University Chest X-ray Collection — 3,337 radiology reports from Indiana University Hospital, made available as an open-access research dataset via the NIH Open-i platform.

License: CC BY-NC-ND 4.0

---

## Notes

The DenseNet121 vision model in this demo uses ImageNet pre-trained weights. For clinical accuracy, it would need fine-tuning on a labelled chest X-ray dataset such as CheXNet (Stanford) or the full NIH ChestX-ray14 dataset. The RAG pipeline and UI are production-ready; the vision component is a prototype.

---

## Author

Harikrishnan Changaramkulath  
M.Sc. Digital Engineering, Otto-von-Guericke-Universität Magdeburg
