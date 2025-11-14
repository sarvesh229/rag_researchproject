# -*- coding: utf-8 -*-
"""
Embedding-only evaluation for Sarvesh's RAG Offer project.

What it does (NO LLMs used):
  - Parses exactly 20 notes from RAG_OFFER_Notes.docx (robust splitter).
  - Loads and chunks all PDFs under ./offers into passages.
  - For each of 5 embedding models, builds a FAISS index and retrieves Top-K per note.
  - Computes retrieval metrics:
      Hit@K (1,3,5,10), Precision@K (1,3,5,10), MRR, nDCG@K (3,5,10),
      Contextual Relevancy (cross-encoder mean score over top-5),
      QTop1Cos (query↔top1 cosine for sanity)
  - Saves:
      results/emb_eval_detailed.csv  (per note × model)
      results/emb_eval_summary.csv   (per model averages)

Dependencies (in your requirements.txt already):
  - langchain, langchain-community, faiss-cpu, pymupdf
  - sentence-transformers, transformers, accelerate, huggingface-hub, torch
  - python-docx, pandas, numpy
  - PLUS: InstructorEmbedding, FlagEmbedding (for e5 / bge)
"""

import os
import re
import math
from typing import List, Dict, Any
import pandas as pd
import torch

from docx import Document

from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Embedding backends
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,  # requires InstructorEmbedding
    HuggingFaceBgeEmbeddings,       # requires FlagEmbedding
)

# ------------------ CONFIG ------------------
OFFERS_DIR   = "offers"
NOTES_DOCX   = "RAG_OFFER_Notes.docx"
TOP_K        = 10
RESULTS_DIR  = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Pick device for the cross-encoder (CPU/GPU). Embeddings run via LangChain backends.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 5 embedders to compare (fixed repo ids)
def m_HF(name):
    return HuggingFaceEmbeddings(model_name=name, encode_kwargs={"normalize_embeddings": True})

def m_E5(name):
    # E5 uses instruction-tuned interface; needs InstructorEmbedding
    return HuggingFaceInstructEmbeddings(
        model_name=name,
        embed_instruction="passage: ",
        query_instruction="query: ",
        encode_kwargs={"normalize_embeddings": True},
    )

def m_BGE(name):
    # BGE has its own LangChain wrapper; needs FlagEmbedding
    return HuggingFaceBgeEmbeddings(
        model_name=name,
        encode_kwargs={"normalize_embeddings": True},
        query_instruction="Represent this question for searching relevant passages: ",
    )

EMBEDDERS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", m_HF),
    "e5":     ("intfloat/e5-base-v2",                      m_E5),
    "bge":    ("BAAI/bge-small-en-v1.5",                   m_BGE),
    "gte":    ("thenlper/gte-small",                       m_HF),  # ← fixed
    "mxbai":  ("mixedbread-ai/mxbai-embed-large-v1",       m_HF),
}

# Cross-encoder reranker used ONLY for scoring contextual relevancy (no reordering)
RERANKER_NAME = "BAAI/bge-reranker-base"
reranker = CrossEncoder(RERANKER_NAME, device=DEVICE)

# ------------------ NOTE PARSER (guarantees exactly N<=20 valid notes) ------------------
NEW_NOTE = re.compile(r"^\s*(?:\d+[\.\)]\s*|note\s*\d+\s*:)\s*", re.I)

def read_notes(docx_path: str, max_notes: int = 20, min_len: int = 25) -> List[str]:
    """
    Robustly parse numbered notes from a .docx file:
      - Recognizes "1.", "2)", "Note 3:", etc.
      - Removes heading like "Notes"
      - Drops very short junk lines (< min_len)
      - Ensures at most `max_notes` notes (trim if needed)
    """
    doc = Document(docx_path)
    paras = [p.text.strip() for p in doc.paragraphs]

    notes: List[str] = []
    cur: List[str] = []

    def flush():
        if not cur:
            return
        txt = " ".join(cur).strip()
        # drop heading+garbage
        if txt and txt.lower() != "notes" and len(txt) >= min_len:
            notes.append(txt)
        cur.clear()

    for line in paras:
        if not line:
            # keep accumulating; don't flush on every blank
            continue

        if NEW_NOTE.match(line):
            flush()
            line = NEW_NOTE.sub("", line, count=1).strip()
            cur = [line] if line else []
        else:
            cur.append(line)
    flush()

    # If we somehow didn't detect numbered lines, fallback to blocks separated by blanks
    if not notes:
        block, blocks = [], []
        for line in paras:
            if line:
                block.append(line)
            elif block:
                blocks.append(" ".join(block).strip())
                block = []
        if block:
            blocks.append(" ".join(block).strip())
        # filter and dedupe
        notes = [b for b in blocks if len(b) >= min_len]
        deduped = []
        for n in notes:
            if not deduped or deduped[-1] != n:
                deduped.append(n)
        notes = deduped

    # Enforce max count (dataset ground truth = 20)
    if len(notes) > max_notes:
        notes = notes[:max_notes]

    return notes

# ------------------ CORPUS LOADING ------------------
def load_offers_as_chunks(offers_dir: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=120)
    all_chunks = []
    for fn in os.listdir(offers_dir):
        if not fn.lower().endswith(".pdf"):
            continue
        path = os.path.join(offers_dir, fn)
        try:
            pages = PyMuPDFLoader(path).load()
        except Exception:
            continue
        # Add filename into content so we can inspect relevance later
        prepared = []
        base = os.path.splitext(fn)[0].lower()
        for d in pages:
            d.metadata = (d.metadata or {}) | {"source_file": fn}
            d.page_content = f"TITLE: {base}\nCONTENT:\n{d.page_content}"
            prepared.append(d)
        chunks = splitter.split_documents(prepared)
        all_chunks.extend(chunks)
    return all_chunks

# ------------------ SIMPLE INTENT → RELEVANCE LABELS ------------------
FURNITURE_TYPES = {
    "bed","sofa","couch","chair","table","desk","wardrobe","cabinet",
    "shelf","bookshelf","dresser","nightstand","stool","bench","futon",
    "recliner","mirror","lamp","ottoman"
}

def extract_intent(note: str) -> Dict[str, Any]:
    t = note.lower()
    types = [w for w in FURNITURE_TYPES if w in t]
    return {"types": types}

def retrieved_labels(intent, retrieved_docs: List[Any]) -> List[bool]:
    """
    Heuristic relevance: a chunk is relevant if it mentions any requested furniture 'type'.
    If you have gold labels (note_id → expected source_file), replace this function accordingly.
    """
    want = set(intent["types"])
    if not want:
        return [False] * len(retrieved_docs)
    flags = []
    for d in retrieved_docs:
        text = d.page_content.lower()
        flags.append(any(w in text for w in want))
    return flags

# ------------------ METRICS ------------------
def precision_at_k(rels: List[bool], k: int) -> float:
    k = min(k, len(rels))
    return (sum(rels[:k]) / k) if k > 0 else 0.0

def hit_at_k(rels: List[bool], k: int) -> float:
    k = min(k, len(rels))
    return float(any(rels[:k])) if k > 0 else 0.0

def mrr(rels: List[bool]) -> float:
    for i, r in enumerate(rels, start=1):
        if r:
            return 1.0 / i
    return 0.0

def ndcg_at_k(gains: List[int], k: int) -> float:
    k = min(k, len(gains))
    if k == 0:
        return 0.0
    dcg = sum(gains[i - 1] / math.log2(i + 1) for i in range(1, k + 1))
    sorted_gains = sorted(gains, reverse=True)
    idcg = sum(sorted_gains[i - 1] / math.log2(i + 1) for i in range(1, k + 1))
    return (dcg / idcg) if idcg > 0 else 0.0

def contextual_relevancy(query: str, docs: List[Any]) -> float:
    """
    Cross-encoder mean score over top-5. Higher = better match between query and retrieved texts.
    (We do NOT use this to reorder — it's only a diagnostic score.)
    """
    pairs = [(query, d.page_content[:1200]) for d in docs[:5]]
    if not pairs:
        return 0.0
    with torch.no_grad():
        scores = reranker.predict(pairs)
    return float(sum(scores) / len(scores))

# ------------------ RETRIEVAL ------------------
def build_faiss(chunks, embedding):
    return FAISS.from_documents(chunks, embedding)

def dense_retrieve_by_query(query: str, vs, k: int) -> List[Any]:
    """
    Standard similarity_search (embeds query internally via the same embedder).
    """
    doc_scores = vs.similarity_search_with_score(query, k=k)
    return [d for d, _ in doc_scores]

def eval_one_model(model_key: str, model_name: str, ctor, notes: List[str], chunks: List[Any]) -> pd.DataFrame:
    print(f"\n=== Building index for {model_key}: {model_name}")
    emb = ctor(model_name)
    vs = build_faiss(chunks, emb)

    rows = []
    for note_idx, note in enumerate(notes, start=1):  # guarantees 1..len(notes)
        intent = extract_intent(note)
        retrieved = dense_retrieve_by_query(note, vs, TOP_K)
        rels = retrieved_labels(intent, retrieved)  # booleans
        gains = [1 if r else 0 for r in rels]

        row_base = {
            "model": model_key,
            "note_id": note_idx,
            "Hit@1":  hit_at_k(rels, 1),
            "Hit@3":  hit_at_k(rels, 3),
            "Hit@5":  hit_at_k(rels, 5),
            "Hit@10": hit_at_k(rels, 10),
            "P@1":    precision_at_k(rels, 1),
            "P@3":    precision_at_k(rels, 3),
            "P@5":    precision_at_k(rels, 5),
            "P@10":   precision_at_k(rels, 10),
            "MRR":    mrr(rels),
            "nDCG@3": ndcg_at_k(gains, 3),
            "nDCG@5": ndcg_at_k(gains, 5),
            "nDCG@10":ndcg_at_k(gains, 10),
            "CtxRel": contextual_relevancy(note, retrieved),
        }

        # Optional: Query ↔ Top1 cosine (same embedder)
        try:
            qv = torch.tensor(emb.embed_query(note), dtype=torch.float32)
            if retrieved:
                tv = torch.tensor(emb.embed_query(retrieved[0].page_content[:1024]), dtype=torch.float32)
                cos = torch.nn.functional.cosine_similarity(qv.unsqueeze(0), tv.unsqueeze(0)).item()
            else:
                cos = float("nan")
            row_base["QTop1Cos"] = cos
        except Exception:
            row_base["QTop1Cos"] = float("nan")

        rows.append(row_base)

    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("model", as_index=False)
          .agg({
              "Hit@1":"mean","Hit@3":"mean","Hit@5":"mean","Hit@10":"mean",
              "P@1":"mean","P@3":"mean","P@5":"mean","P@10":"mean",
              "MRR":"mean",
              "nDCG@3":"mean","nDCG@5":"mean","nDCG@10":"mean",
              "CtxRel":"mean",
              "QTop1Cos":"mean",
          })
          .sort_values(["Hit@5","MRR"], ascending=False)
    )
    return agg

def main():
    # --- Load notes (expect 20) ---
    notes = read_notes(NOTES_DOCX, max_notes=20, min_len=25)
    print(f"Parsed notes: {len(notes)} (expect 20)")
    for i, n in enumerate(notes, 1):
        preview = (n[:90] + "...") if len(n) > 90 else n
        print(f"[{i:02}] {preview}")

    # --- Load and chunk offers corpus ---
    print("\nLoading and chunking PDFs from ./offers ...")
    chunks = load_offers_as_chunks(OFFERS_DIR)
    print(f"Loaded {len(chunks)} chunks.")

    # --- Evaluate all models ---
    all_rows = []
    for key, (name, ctor) in EMBEDDERS.items():
        df = eval_one_model(key, name, ctor, notes, chunks)
        all_rows.append(df)

    detailed = pd.concat(all_rows, ignore_index=True)
    summary  = summarize(detailed)

    # --- Save outputs ---
    detailed_path = os.path.join(RESULTS_DIR, "emb_eval_detailed.csv")
    summary_path  = os.path.join(RESULTS_DIR, "emb_eval_summary.csv")
    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(" -", detailed_path)
    print(" -", summary_path)
    print("\nSummary:\n", summary.to_string(index=False))

if __name__ == "__main__":
    main()
