import os, glob, time
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
import faiss

# ---------- KONFIGURASI (VERIFIKASI ID model di Google AI Studio) ----------
EMBED_MODEL = "gemini-embedding-001"   # model embedding terkini; cek di AI Studio
CHAT_MODEL  = "gemini-2.5-flash"       # pilih model Flash yang gratis di free tier
CHUNK_WORDS = 250
CHUNK_OVERLAP = 50
TOP_K = 4

# SDK baru: buat client sekali, lalu panggil client.models.*
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# ---------- BACA & POTONG DOKUMEN ----------
def read_docs(folder="docs"):
    items = []
    for path in glob.glob(os.path.join(folder, "*")):
        name = os.path.basename(path)
        if path.lower().endswith(".pdf"):
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    items.append((f"{name} (hal. {i})", text))
        elif path.lower().endswith(".txt"):
            with open(path, encoding="utf-8") as f:
                items.append((name, f.read()))
    return items

def chunk(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i+size]))
        i += size - overlap
    return out

# ---------- EMBEDDING (SDK baru: client.models.embed_content) ----------
# task = "RETRIEVAL_DOCUMENT" saat indexing, "RETRIEVAL_QUERY" saat bertanya.
# Jika task_type memicu error, hapus argumen config= (embedding tetap jalan).
def embed_texts(texts, task, batch=100):
    out = []
    for i in range(0, len(texts), batch):
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts[i:i+batch],
            config=types.EmbedContentConfig(task_type=task),
        )
        out.extend(e.values for e in resp.embeddings)
        time.sleep(0.05)  # jaga-jaga rate limit saat indexing
    return np.array(out, dtype="float32")

# ---------- BANGUN INDEX (di-cache: hanya sekali) ----------
@st.cache_resource(show_spinner="Membangun indeks pengetahuan...")
def build_index():
    docs = read_docs()
    meta = []            # (sumber, teks_chunk)
    for source, text in docs:
        for c in chunk(text):
            meta.append((source, c))
    mat = embed_texts([m[1] for m in meta], task="RETRIEVAL_DOCUMENT")
    faiss.normalize_L2(mat)              # cosine via inner product
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    return index, meta

# ---------- RETRIEVAL ----------
def retrieve(query, index, meta, k=TOP_K):
    q = embed_texts([query], task="RETRIEVAL_QUERY")
    faiss.normalize_L2(q)
    scores, idx = index.search(q, k)
    return [(meta[i][0], meta[i][1], float(scores[0][j]))
            for j, i in enumerate(idx[0])]

# ---------- GENERATION (SDK baru: client.models.generate_content) ----------
def generate_answer(query, contexts):
    blok = "\n\n".join(
        f"[Sumber {n}] ({src})\n{txt}" for n, (src, txt, _) in enumerate(contexts, 1)
    )
    prompt = f"""Anda asisten pengetahuan tata kelola (GCG). Jawab HANYA berdasarkan konteks di bawah.
Jika informasi tidak ada di konteks, katakan "Informasi tidak ditemukan pada dokumen yang tersedia."
Selalu cantumkan rujukan dalam bentuk [Sumber N] di akhir kalimat yang relevan. Jawab dalam Bahasa Indonesia.

KONTEKS:
{blok}

PERTANYAAN: {query}

JAWABAN:"""
    resp = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return resp.text

# ---------- UI ----------
st.title("Governance Knowledge Management System — Demo")
st.caption("Prototipe RAG. Jawaban selalu merujuk ke dokumen sumber. Hanya dokumen publik.")

index, meta = build_index()
q = st.chat_input("Tanyakan sesuatu tentang GCG, gratifikasi, benturan kepentingan, dst.")
if q:
    st.chat_message("user").write(q)
    hits = retrieve(q, index, meta)
    answer = generate_answer(q, hits)
    with st.chat_message("assistant"):
        st.write(answer)
        with st.expander("Lihat sumber yang diambil"):
            for n, (src, txt, score) in enumerate(hits, 1):
                st.markdown(f"**[Sumber {n}]** {src}  ·  skor {score:.2f}")
                st.write(txt[:400] + ("..." if len(txt) > 400 else ""))
