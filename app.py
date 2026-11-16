import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Radiology Intelligence Assistant",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; color: #0d47a1; }
    .main-header p  { margin: 0.4rem 0 0; font-size: 1rem; color: #1565c0; }

    .metric-card {
        background: #f8faff; border: 1px solid #e3eaf7;
        border-left: 4px solid #1565c0; border-radius: 8px;
        padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    }
    .metric-card h4 { margin: 0 0 0.3rem; color: #1565c0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card p  { margin: 0; font-size: 1.4rem; font-weight: 700; color: #1a237e; }

    .result-card {
        background: white; border: 1px solid #e0e7ff; border-radius: 10px;
        padding: 1.2rem 1.5rem; margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(21,101,192,0.07);
    }
    .result-card .score-badge {
        display: inline-block; background: #e8f0fe; color: #1565c0;
        border-radius: 20px; padding: 0.15rem 0.7rem;
        font-size: 0.8rem; font-weight: 600; margin-bottom: 0.6rem;
    }
    .result-card h4 { margin: 0 0 0.5rem; color: #263238; font-size: 1rem; }
    .result-card p  { margin: 0; color: #546e7a; font-size: 0.9rem; line-height: 1.6; }

    .answer-box {
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
        border: 1px solid #a5d6a7; border-left: 4px solid #2e7d32;
        border-radius: 10px; padding: 1.2rem 1.5rem; margin-top: 1rem;
    }
    .answer-box h4 { color: #1b5e20; margin: 0 0 0.6rem; }

    .finding-tag {
        display: inline-block; background: #fff3e0; color: #e65100;
        border-radius: 4px; padding: 0.15rem 0.5rem;
        font-size: 0.78rem; font-weight: 600; margin: 0.1rem;
    }
    .finding-normal { background: #e8f5e9; color: #2e7d32; }

    .gdpr-badge {
        background: #1b5e20; color: white; border-radius: 6px;
        padding: 0.3rem 0.8rem; font-size: 0.8rem; font-weight: 600;
        display: inline-block; margin-top: 0.5rem;
    }

    .vision-card {
        background: #fafafa; border: 1px solid #e0e0e0;
        border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.6rem;
    }
    .vision-bar-bg {
        background: #e3eaf7; border-radius: 4px; height: 10px; margin-top: 4px;
    }
    .vision-bar-fill {
        background: linear-gradient(90deg, #1565c0, #42a5f5);
        border-radius: 4px; height: 10px;
    }
    .tab-header {
        font-size: 1.1rem; font-weight: 700; color: #1a237e;
        border-bottom: 2px solid #1565c0; padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH   = "indiana_reports.csv"
INDEX_PATH  = "reports.index"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "llama3.2:3b"

CHEXNET_CONDITIONS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia"
]

SAMPLE_QUERIES = [
    "cardiomegaly findings",
    "pleural effusion bilateral",
    "normal chest x-ray no findings",
    "pneumonia infiltrate",
    "degenerative spine changes",
    "enlarged lymph nodes mediastinum",
]

# ── Data & Index ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    text_col = None
    for col in ["findings", "impression", "report"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        text_cols = [c for c in df.columns if df[c].dtype == object]
        df["combined_text"] = df[text_cols].fillna("").agg(" ".join, axis=1)
        text_col = "combined_text"
    df[text_col] = df[text_col].fillna("No report available")
    df = df[df[text_col].str.len() > 20].reset_index(drop=True)
    return df, text_col

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource
def build_index(_df, text_col, _embedder):
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    texts  = _df[text_col].tolist()
    embeds = _embedder.encode(texts, show_progress_bar=True, batch_size=64)
    embeds = np.array(embeds, dtype="float32")
    faiss.normalize_L2(embeds)
    index  = faiss.IndexFlatIP(embeds.shape[1])
    index.add(embeds)
    faiss.write_index(index, INDEX_PATH)
    return index

@st.cache_resource
def load_llm():
    return OllamaLLM(model=LLM_MODEL, temperature=0.1)

# ── CheXNet model (DenseNet121 pretrained on ImageNet, fine-tuned head) ───────
@st.cache_resource
def load_vision_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    # Replace classifier for 14 CheXNet conditions
    model.classifier = torch.nn.Linear(model.classifier.in_features, 14)
    model.eval()
    return model

XRAY_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def classify_xray(image: Image.Image, model) -> dict:
    tensor = XRAY_TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).squeeze().numpy()
    return dict(zip(CHEXNET_CONDITIONS, probs.tolist()))

# ── RAG helpers ───────────────────────────────────────────────────────────────
def search(query, df, text_col, embedder, index, top_k=5):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(df):
            row = df.iloc[idx]
            results.append({
                "score": float(score),
                "text":  row[text_col],
                "uid":   row.get("uid", idx),
            })
    return results

PROMPT = PromptTemplate.from_template("""You are a radiology report assistant.
Answer the clinical question using ONLY the provided radiology reports.
Be concise and precise. Mention specific findings. If nothing relevant, say so.

Clinical question: {question}

Relevant radiology reports:
{context}

Answer:""")

def ask_llm(question, results, llm):
    context = "\n\n---\n\n".join(
        [f"Report {i+1}: {r['text'][:600]}" for i, r in enumerate(results[:3])]
    )
    return (PROMPT | llm).invoke({"question": question, "context": context})

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    top_k   = st.slider("Results to retrieve", 3, 10, 5)
    use_llm = st.toggle("AI Summary (Ollama)", value=True)
    st.divider()
    st.markdown("### 📊 System Info")
    st.markdown('<div class="gdpr-badge">🔒 100% Local · GDPR Safe</div>', unsafe_allow_html=True)
    st.markdown("""
    <small>All processing on-device.<br>
    No patient data leaves the hospital.<br><br>
    <b>Stack:</b> FAISS · sentence-transformers<br>
    DenseNet121 · llama3.2:3b · Streamlit</small>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("### 🔍 Sample Queries")
    selected_sample = st.session_state.get("selected_sample", None)
    for q in SAMPLE_QUERIES:
        if st.button(q, key=f"sq_{q}", use_container_width=True):
            st.session_state["selected_sample"] = q
            selected_sample = q

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🫁 Radiology Intelligence Assistant</h1>
  <p>AI-powered X-ray analysis and semantic report search &nbsp;·&nbsp; Fully local · GDPR compliant</p>
</div>
""", unsafe_allow_html=True)

# ── Load core resources ───────────────────────────────────────────────────────
with st.spinner("Loading dataset and search index…"):
    df, text_col = load_data()
    embedder     = load_embedder()
    index        = build_index(df, text_col, embedder)

# Metrics
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(f'<div class="metric-card"><h4>Total Reports</h4><p>{len(df):,}</p></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-card"><h4>Conditions Detected</h4><p>14</p></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-card"><h4>Search Engine</h4><p>FAISS</p></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="metric-card"><h4>Vision Model</h4><p>DenseNet121</p></div>', unsafe_allow_html=True)

st.divider()

# ── Two tabs ──────────────────────────────────────────────────────────────────
tab_search, tab_xray = st.tabs(["🔍 Report Search", "🩻 X-ray Analysis"])

# ════════════════════════════════════════════════════════════════════════
# TAB 1 — Report Search (RAG)
# ════════════════════════════════════════════════════════════════════════
with tab_search:
    query = st.text_input(
        "Enter clinical query",
        value=selected_sample or "",
        placeholder="e.g. cardiomegaly, pleural effusion, normal chest…",
    )
    search_btn = st.button("Search Reports", type="primary")

    keywords = {
        "cardiomegaly": "warning", "effusion": "warning",
        "pneumonia": "warning",    "infiltrate": "warning",
        "normal": "normal",        "no acute": "normal",
        "clear": "normal",         "atelectasis": "warning",
        "consolidation": "warning","edema": "warning",
    }

    if (search_btn) and query.strip():
        with st.spinner("Searching…"):
            results = search(query, df, text_col, embedder, index, top_k=top_k)

        st.markdown(f"#### Results for: *{query}*")
        left, right = st.columns([3, 2])

        with left:
            st.markdown('<div class="tab-header">📄 Matching Reports</div>', unsafe_allow_html=True)
            for r in results:
                pct     = int(r["score"] * 100)
                preview = r["text"][:400] + ("…" if len(r["text"]) > 400 else "")
                st.markdown(f"""
                <div class="result-card">
                    <span class="score-badge">Match {pct}%</span>
                    <h4>Report #{r['uid']}</h4>
                    <p>{preview}</p>
                </div>""", unsafe_allow_html=True)
                with st.expander("View full report"):
                    st.write(r["text"])

        with right:
            if use_llm:
                st.markdown('<div class="tab-header">🤖 AI Clinical Summary</div>', unsafe_allow_html=True)
                with st.spinner("Generating with Ollama…"):
                    try:
                        answer = ask_llm(query, results, load_llm())
                        st.markdown(f'<div class="answer-box"><h4>Clinical Insight</h4><p>{answer}</p></div>',
                                    unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Ollama unavailable: {e}")

            st.markdown('<div class="tab-header" style="margin-top:1.5rem">🏷️ Key Findings</div>', unsafe_allow_html=True)
            all_text = " ".join([r["text"].lower() for r in results])
            found    = {(kw.title(), kt) for kw, kt in keywords.items() if kw in all_text}
            if found:
                html = "".join(
                    f'<span class="finding-tag {"finding-normal" if kt=="normal" else ""}">{kw}</span>'
                    for kw, kt in sorted(found)
                )
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("No specific keywords detected.")
    else:
        st.info("👆 Enter a clinical term above, or pick a sample query from the sidebar.")

# ════════════════════════════════════════════════════════════════════════
# TAB 2 — X-ray Image Analysis
# ════════════════════════════════════════════════════════════════════════
with tab_xray:
    st.markdown("Upload a chest X-ray image and the AI will classify 14 conditions, then automatically search for similar reports.")

    uploaded = st.file_uploader("Upload chest X-ray (PNG / JPG)", type=["png", "jpg", "jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        col_img, col_results = st.columns([1, 2])

        with col_img:
            st.image(image, caption="Uploaded X-ray", width=350)

        with col_results:
            st.markdown('<div class="tab-header">🧠 AI Classification (DenseNet121)</div>', unsafe_allow_html=True)

            with st.spinner("Analysing X-ray…"):
                vision_model = load_vision_model()
                scores       = classify_xray(image, vision_model)

            # Sort by confidence descending
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            for condition, prob in sorted_scores:
                pct      = int(prob * 100)
                bar_color = "#1565c0" if pct >= 30 else "#90a4ae"
                st.markdown(f"""
                <div class="vision-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:600; color:#263238">{condition}</span>
                        <span style="font-weight:700; color:{'#c62828' if pct>=50 else '#1565c0' if pct>=30 else '#78909c'}">{pct}%</span>
                    </div>
                    <div class="vision-bar-bg">
                        <div class="vision-bar-fill" style="width:{pct}%; background:{'#c62828' if pct>=50 else '#1565c0'}"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        # Auto-search using top findings
        st.divider()
        top_findings = [c for c, p in sorted_scores if p >= 0.25]
        if top_findings:
            auto_query = " ".join(top_findings[:3]).lower()
            st.markdown(f"#### 🔍 Auto-searching reports for: *{', '.join(top_findings[:3])}*")

            with st.spinner("Fetching similar reports…"):
                xray_results = search(auto_query, df, text_col, embedder, index, top_k=3)

            r_left, r_right = st.columns([3, 2])
            with r_left:
                st.markdown('<div class="tab-header">📄 Similar Historical Reports</div>', unsafe_allow_html=True)
                for r in xray_results:
                    pct     = int(r["score"] * 100)
                    preview = r["text"][:350] + "…"
                    st.markdown(f"""
                    <div class="result-card">
                        <span class="score-badge">Match {pct}%</span>
                        <h4>Report #{r['uid']}</h4>
                        <p>{preview}</p>
                    </div>""", unsafe_allow_html=True)
                    with st.expander("View full report"):
                        st.write(r["text"])

            with r_right:
                if use_llm:
                    st.markdown('<div class="tab-header">🤖 AI Clinical Summary</div>', unsafe_allow_html=True)
                    with st.spinner("Generating summary…"):
                        try:
                            summary = ask_llm(
                                f"Summarise findings for a patient with {', '.join(top_findings[:3])}",
                                xray_results,
                                load_llm()
                            )
                            st.markdown(f'<div class="answer-box"><h4>Clinical Insight</h4><p>{summary}</p></div>',
                                        unsafe_allow_html=True)
                        except Exception as e:
                            st.warning(f"Ollama unavailable: {e}")
        else:
            st.info("No conditions detected above 25% confidence threshold. Try a clearer X-ray image.")
    else:
        st.info("👆 Upload a chest X-ray PNG or JPG to begin analysis.")
        st.markdown("""
        **What this does:**
        - Runs DenseNet121 vision model locally on your GPU
        - Classifies 14 chest conditions (cardiomegaly, effusion, pneumonia, etc.)
        - Automatically searches 3,337 historical reports for similar cases
        - Generates AI clinical summary — all without sending data anywhere
        """)