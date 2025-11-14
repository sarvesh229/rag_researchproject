# -*- coding: utf-8 -*-
"""
Offer generator (updated):
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Retrieval: FAISS (dense) + BM25Retriever (lexical) + literal-phrase booster + rerank
- Extraction: LLM to JSON with a robust rule-based fallback
- Dedupe: collapse near-identical items to a single line
- Output: PDF with price column right-aligned (and never blank)

Run:
  python working.py
"""

import os, re, json, uuid
import fitz
import requests
from datetime import datetime
from dotenv import load_dotenv
import difflib
from typing import List, Dict, Any, Tuple, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DEBUG = True  # set False to reduce console logs

# ----------------------------- #
# Together.ai Chat Completion
# ----------------------------- #
def together_chat(messages, model="mistralai/Mistral-7B-Instruct-v0.3", temperature=0.2, max_tokens=700):
    load_dotenv()
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY not set in environment/.env")
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise Exception(f"API Error: {r.status_code} - {r.text}")
    return r.json()["choices"][0]["message"]["content"]

# ----------------------------- #
# Load Offer Documents (chunked + whole-file)
# ----------------------------- #
def load_offer_documents(folder_path):
    print(f"📂 Loading from '{folder_path}'...")
    chunked_docs, full_docs = [], []
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=120)

    for file in os.listdir(folder_path):
        if not file.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder_path, file)
        loader = PyMuPDFLoader(path)
        raw_docs = loader.load()  # per-page

        base = os.path.splitext(file)[0].lower()
        category_hint = base.replace("-", " ").replace("_", " ")
        first_token = category_hint.split()[0] if category_hint else ""

        # Whole-file doc for literal phrase search
        whole_text = "\n".join(d.page_content for d in raw_docs)
        whole = raw_docs[0].copy()
        whole.page_content = whole_text
        whole.metadata = whole.metadata or {}
        whole.metadata.update({"source_file": file, "category": first_token})
        full_docs.append(whole)

        # Chunked docs with identity prefix
        prepared = []
        for d in raw_docs:
            d.metadata = d.metadata or {}
            d.metadata.update({"source_file": file, "category": first_token})
            d.page_content = f"TITLE: {base}\nCONTENT:\n{d.page_content}"
            prepared.append(d)
        chunked_docs.extend(splitter.split_documents(prepared))

    if not chunked_docs:
        raise RuntimeError(f"No PDFs found in '{folder_path}'.")
    print(f"✅ Loaded {len(chunked_docs)} chunks from {len(full_docs)} PDFs.")
    return chunked_docs, full_docs

# ----------------------------- #
# Create FAISS (dense) + BM25 (lexical retriever)
# ----------------------------- #
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def create_vectorstores(docs, faiss_path="faiss_index"):
    embedding = HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = FAISS.from_documents(docs, embedding)
    vs.save_local(faiss_path)
    bm25 = BM25Retriever.from_documents(docs)
    return vs, bm25

# ----------------------------- #
# Query understanding + filters
# ----------------------------- #
FURNITURE_TYPES = [
    "bed","sofa","couch","chair","table","desk","wardrobe",
    "cabinet","shelf","bookshelf","dresser","nightstand","stool","bench"
]
COLORS = ["black","white","brown","oak","walnut","grey","gray","beige","natural"]
MATERIALS = ["wood","wooden","engineered wood","mango","acacia","teak","metal","steel","iron","plywood"]
FOLD_SYNS = ("foldable","folding","collapsible","portable","fold-away","fold away","fold up")

def parse_intent(q: str):
    ql = q.lower()
    ftype = next((t for t in FURNITURE_TYPES if t in ql), None)
    color = next((c for c in COLORS if c in ql), None)
    material = next((m for m in MATERIALS if m in ql), None)
    return {"type": ftype, "color": color, "material": material, "ql": ql}

def expand_query(q: str, intent):
    ql = q.lower()
    parts = [ql]
    if intent["type"] == "table":
        parts += ["table", "tables", "table"]
        if "fold" in ql or "table" in ql:
            parts += list(FOLD_SYNS)
    if "light" in ql:
        parts += ["lightweight", "light weight", "light-weight", "portable"]
    return " ".join(parts)

def build_phrases(intent):
    if intent["type"] == "table":
        return ["foldable table", "folding table", "collapsible table", "portable folding table"]
    return []

def metadata_filter(docs, intent):
    if not intent["type"]:
        return docs
    t = intent["type"]
    keep = []
    for d in docs:
        cat = (d.metadata or {}).get("category", "")
        text = d.page_content.lower()
        if t in cat or f" {t} " in (" " + text + " ") or text.startswith(f"title: {t}"):
            keep.append(d)
    return keep or docs

def keyword_gate(docs, intent):
    if intent["type"] != "table":
        return docs
    required_any = FOLD_SYNS
    gated = []
    for d in docs:
        t = d.page_content.lower()
        if "table" in t and any(w in t for w in required_any):
            gated.append(d)
    return gated or docs

# ----------------------------- #
# Hybrid search + fusion
# ----------------------------- #
def hybrid_search(query, faiss_store, bm25_retriever, k=18):
    dense = faiss_store.similarity_search_with_score(query, k=k*3)  # overfetch
    sparse = bm25_retriever.get_relevant_documents(query)[:k*3]

    scores, pool = {}, {}
    for i, (d, _) in enumerate(dense):
        pool[id(d)] = d
        scores[id(d)] = scores.get(id(d), 0.0) + 1.0 / (1 + i)
    for i, d in enumerate(sparse):
        pool[id(d)] = d
        scores[id(d)] = scores.get(id(d), 0.0) + 1.0 / (1 + i)

    ranked = sorted(pool.values(), key=lambda doc: scores[id(doc)], reverse=True)
    return ranked[:k]

# ----------------------------- #
# Literal phrase booster
# ----------------------------- #
def literal_phrase_hits(full_docs, phrases):
    hits = []
    for d in full_docs:
        txt = d.page_content.lower()
        if any(p in txt for p in phrases):
            hits.append(d)
    return hits

def promote_hits(hit_files, all_chunks, k_each=6):
    if not hit_files:
        return []
    picks = []
    for d in all_chunks:
        if (d.metadata or {}).get("source_file") in hit_files:
            picks.append(d)
            if len(picks) >= k_each * len(hit_files):
                break
    return picks

# ----------------------------- #
# (Optional) Cross-encoder rerank
# ----------------------------- #
def rerank(query, docs, top_k=3):
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, d.page_content[:1200]) for d in docs]
        scores = ce.predict(pairs)
        ranked = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)][:top_k]
        top_score = float(sorted(scores, reverse=True)[0]) if len(scores) else None
        return ranked, top_score
    except Exception:
        return docs[:top_k], None

# ----------------------------- #
# Match logic
# ----------------------------- #
def match_product(user_query, faiss_store, bm25_retriever, all_chunks, full_docs):
    intent = parse_intent(user_query)
    phrases = build_phrases(intent)

    # 1) Literal phrase booster at whole-file level
    literal_hits = literal_phrase_hits(full_docs, [p for p in phrases if p])
    boosted_files = {d.metadata.get("source_file") for d in literal_hits}
    boosted_chunks = promote_hits(boosted_files, all_chunks, k_each=6)

    # 2) Hybrid search (expanded query)
    expanded_q = expand_query(user_query, intent)
    candidates = hybrid_search(expanded_q, faiss_store, bm25_retriever, k=24)

    # 3) Merge + filters
    merged, seen = [], set()
    for d in boosted_chunks + candidates:
        key = (d.metadata.get("source_file"), d.page_content[:150])
        if key not in seen:
            merged.append(d); seen.add(key)

    merged = metadata_filter(merged, intent)
    merged = keyword_gate(merged, intent)

    # 4) Rerank
    top_docs, top_score = rerank(user_query, merged, top_k=3)
    if not top_docs:
        return "none", None

    label = "similar"
    if top_score is not None and top_score >= 0.55:
        label = "exact"
    if boosted_files:
        label = "exact"

    if intent["type"]:
        has_type = any(intent["type"] in d.page_content.lower() for d in top_docs)
        if not has_type:
            label = "none"

    if DEBUG:
        print("\n🗂 Matched files:")
        for d in top_docs:
            print(" -", d.metadata.get("source_file"))
    return label, top_docs

# ----------------------------- #
# LLM JSON extraction
# ----------------------------- #
JSON_SYSTEM = "You extract product line items from catalog text. Return STRICT JSON ONLY. No markdown, no comments."

def extract_items_via_llm(user_query, context, max_items=3):
    schema_hint = {
        "items": [
            {"pos":1,"qty":1,"unit":"pc","name":"Product name","material":"Wood",
             "dimensions":"200x160x40 cm","brand":"BrandName","description":"One-sentence description.",
             "price":999.99,"currency":"EUR"}
        ]
    }
    prompt = (
        "USER REQUEST:\n"
        f"{user_query}\n\n"
        "PRODUCT CANDIDATES (raw text, may contain multiple items):\n"
        f"{context}\n\n"
        f"Select up to {max_items} best-matching items. "
        "Always fill a concrete 'name' copied from the text. "
        "Use numbers for prices if present; if no price is present, set price to null and currency to 'EUR'. "
        "Keep strings short (no line breaks). "
        "Output JSON exactly matching this schema:\n"
        f"{json.dumps(schema_hint, ensure_ascii=False)}"
    )
    out = together_chat(
        [{"role":"system","content":JSON_SYSTEM},{"role":"user","content":prompt}],
        temperature=0.0, max_tokens=500
    ).strip()
    out = re.sub(r"^```(?:json)?\s*|\s*```$", "", out, flags=re.IGNORECASE|re.DOTALL)
    try:
        data = json.loads(out)
        assert "items" in data and isinstance(data["items"], list)
        if DEBUG: print("LLM JSON:", json.dumps(data, indent=2)[:400], "...")
        return data
    except Exception as e:
        if DEBUG: print("LLM JSON parse failed:", e, "\nRaw:", out[:400], "...")
        return {"items":[]}

# ----------------------------- #
# Rule-based regex fallback (intent-aware title picking)
# ----------------------------- #
PRICE_RE = re.compile(r'(?P<cur>€|EUR|\$|USD)\s*(?P<num>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)')
BAD_PREFIXES = ("material", "dimensions", "brand", "pos", "qty", "description", "price", "net", "vat", "total")

def norm_amount(num: str) -> Optional[float]:
    if num is None: return None
    n = num.replace(" ", "")
    if n.count(",") > 0 and n.count(".") > 0:
        if n.rfind(",") > n.rfind("."):
            n = n.replace(".", "").replace(",", ".")
        else:
            n = n.replace(",", "")
    else:
        n = n.replace(",", "")
    try:
        return float(n)
    except Exception:
        return None

def _is_bad_prefix(s: str) -> bool:
    low = s.lower().strip()
    return any(low.startswith(p + ":") for p in BAD_PREFIXES)

def _title_score(line: str, intent_type: Optional[str]) -> int:
    s = line.strip()
    low = s.lower()
    if _is_bad_prefix(s): return -10
    if len(s) < 6 or len(s) > 120: return -5
    if re.search(r"^\s*(€|\$|\d)", s): return -5
    score = 0
    if intent_type and intent_type in low: score += 4
    if intent_type == "table" and any(w in low for w in FOLD_SYNS): score += 3
    if not s.endswith("."): score += 1
    cap_tokens = sum(1 for w in re.findall(r"[A-Za-z]+", s) if w[0].isupper())
    score += min(cap_tokens, 4)
    if "," in s or " - " in s: score += 1
    return score

def extract_items_rule_based(context: str, intent: dict, max_items=3):
    lines = [l.strip() for l in context.splitlines() if l.strip()]
    items = []

    for i, line in enumerate(lines):
        m = PRICE_RE.search(line)
        if not m:
            continue

        # best title-like line near the price
        window_idx = range(max(0, i-8), min(len(lines), i+9))
        best_line, best_sc = None, -999
        for k in window_idx:
            cand = lines[k]
            sc = _title_score(cand, intent.get("type"))
            if sc > best_sc:
                best_sc, best_line = sc, cand

        if best_sc < 1:
            for j in range(1, 7):
                k = i - j
                if k < 0: break
                cand = lines[k]
                if _is_bad_prefix(cand):
                    continue
                if len(cand.split()) >= 3 and not cand.endswith("."):
                    best_line = cand
                    break

        name = (best_line or "Item").strip()
        amount = norm_amount(m.group("num"))
        cur = "EUR" if m.group("cur") in ("€","EUR") else "USD"

        material = dimensions = brand = None
        for j in range(1, 7):
            k = i + j
            if k >= len(lines): break
            low = lines[k].lower()
            if low.startswith("material"):
                material = lines[k].split(":",1)[-1].strip()
            elif low.startswith("dimensions"):
                dimensions = lines[k].split(":",1)[-1].strip()
            elif low.startswith("brand"):
                brand = lines[k].split(":",1)[-1].strip()

        items.append({
            "pos": len(items)+1,
            "qty": 1, "unit":"pc",
            "name": name,
            "material": material, "dimensions": dimensions, "brand": brand,
            "description": "",
            "price": amount, "currency": cur
        })
        if len(items) >= max_items:
            break

    if DEBUG and items:
        print("Regex items (after title scoring):", json.dumps(items, indent=2))
    return items

def extract_items(user_query, context, max_items=3):
    # 1) Try LLM JSON
    data = extract_items_via_llm(user_query, context, max_items=max_items)
    items = data.get("items", [])
    good = [it for it in items if it.get("name") and it.get("name").lower() != "selected item"]

    # 2) If LLM weak or no price, use rule-based
    if not good or all(it.get("price") in (None, "null") for it in good):
        intent = parse_intent(user_query)
        rb = extract_items_rule_based(context, intent, max_items=max_items)
        if rb:
            return rb

    return good or [{"pos":1,"qty":1,"unit":"pc","name":"Selected Item","price":None,"currency":"EUR"}]

# ----------------------------- #
# DEDUPLICATION of same/near-same items
# ----------------------------- #
def _normalize_title(s: str) -> str:
    if not s: return ""
    s = s.lower()
    s = re.sub(r'\b(set|pack|bundle)\s*(of)?\s*\d+\b', '', s)
    s = re.sub(r'\b\d+(\.\d+)?\s*(x|\u00D7)\s*\d+(\.\d+)?(\s*(x|\u00D7)\s*\d+(\.\d+)?)?\s*(cm|mm|inches|inch|")\b', '', s)
    s = re.sub(r'\b\d+(\.\d+)?\s*("|in|inch|inches|cm|mm)\b', '', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    stop = {'brand','store','model','series','with','and','for','the','a','an','of'}
    tokens = [t for t in s.split() if t not in stop]
    return ' '.join(tokens)

def _similar(a: str, b: str, thresh: float = 0.86) -> bool:
    aa, bb = _normalize_title(a), _normalize_title(b)
    if not aa or not bb:
        return False
    return difflib.SequenceMatcher(None, aa, bb).ratio() >= thresh

def dedupe_items_by_name(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: List[List[Dict[str, Any]]] = []
    for it in items:
        name = (it.get("name") or "").strip()
        placed = False
        for g in groups:
            if _similar(name, g[0].get("name","")):
                g.append(it)
                placed = True
                break
        if not placed:
            groups.append([it])

    deduped = []
    for g in groups:
        def score(x):
            has_price = 1 if isinstance(x.get("price"), (int,float)) else 0
            filled = sum(1 for k in ("material","dimensions","brand","description") if x.get(k))
            title_len = len(_normalize_title(x.get("name","")))
            return (has_price, filled, title_len)
        keep = sorted(g, key=score, reverse=True)[0]
        keep["pos"] = len(deduped) + 1
        deduped.append(keep)
    return deduped

# ----------------------------- #
# PDF builder with right-aligned price
# ----------------------------- #
def currency_symbol(cur: str) -> str:
    if not cur: return "€"
    c = cur.upper()
    return {"USD":"$", "EUR":"€", "GBP":"£", "INR":"₹"}.get(c, "€")

def fmt_price(amount, sym):
    if amount is None: return "TBD"
    return f"{sym}{amount:,.2f}"

def build_and_render_offer_pdf(offer_id, customer_name, customer_no, items, vat_rate=0.19, default_currency="EUR", filename="offer.pdf"):
    import fitz
    from datetime import datetime

    # Page + layout
    W, H = 595, 842              # A4
    LM, TM, RM, BM = 36, 36, 36, 36
    y = TM
    line = 16
    fs_h1, fs_h2, fs, fs_small = 18, 12, 11, 10

    sym = currency_symbol((items[0].get("currency") if items else None) or default_currency)
    today = datetime.now().strftime("%d.%m.%Y")

    # Columns
    col_pos = LM
    col_qty = col_pos + 40
    col_desc = col_qty + 44
    col_price = W - RM  # right edge to align price

    # Helpers — NO fontname anywhere
    def draw_text(page, x, y, text, size=fs):
        page.insert_text(fitz.Point(x, y), text, fontsize=size, color=(0, 0, 0))

    def draw_right(page, x_right, y, text, size=fs):
        rect = fitz.Rect(LM, y - 12, x_right, y + 20)
        try:
            page.insert_textbox(rect, text, fontsize=size, align=fitz.TEXT_ALIGN_RIGHT)
        except TypeError:
            # very old PyMuPDF with no 'align' kwarg: crude right-align fallback
            pad = max(0, int((x_right - (col_desc + 10)) / 6) - len(text))
            draw_text(page, col_desc + 10, y, " " * pad + text, size=size)

    def wrap_to_width(text, max_width_pt):
        # approx char-per-line calc (kept simple for robustness)
        approx_char = int(max_width_pt / 6.0)
        words = text.split()
        out, cur = [], ""
        for w in words:
            cand = (cur + " " + w).strip()
            if len(cand) > approx_char:
                if cur:
                    out.append(cur)
                cur = w
            else:
                cur = cand
        if cur:
            out.append(cur)
        return out

    # Create document
    doc = fitz.open()
    page = doc.new_page(width=W, height=H)

    # Header
    draw_text(page, LM, y, f"Offer ID: {offer_id}", fs_h1); y += line * 1.6
    draw_text(page, LM, y, f"Customer: {customer_name}", fs_h2); y += line
    draw_text(page, LM, y, f"Customer No: {customer_no}", fs_h2); y += line
    draw_text(page, LM, y, f"Date: {today}", fs_h2); y += line * 1.4
    draw_text(page, LM, y, "We are pleased to present you with an offer for the following items:", fs); y += line * 1.3

    # Table header
    page.draw_rect(fitz.Rect(LM, y - 10, W - RM, y - 9), color=(0, 0, 0), fill=(0, 0, 0))
    draw_text(page, col_pos, y, "Pos", fs_small)
    draw_text(page, col_qty, y, "Qty", fs_small)
    draw_text(page, col_desc, y, "Description", fs_small)
    draw_right(page, col_price, y, "Price", fs_small); y += line

    # Items
    net = 0.0
    for idx, it in enumerate(items, 1):
        pos = f"{idx:03}"
        qty = f"{int(it.get('qty', 1))}"
        unit = it.get("unit", "pc")
        name = (it.get("name") or "Item").strip()
        material = it.get("material")
        dims = it.get("dimensions")
        brand = it.get("brand")
        desc = it.get("description") or ""
        price = it.get("price")
        price_str = fmt_price(price, sym)

        # Left columns
        draw_text(page, col_pos, y, pos)
        draw_text(page, col_qty, y, qty)
        draw_text(page, col_qty, y + line * 0.9, unit, fs_small)

        # Description (wrapped)
        lines = [name]
        if material: lines.append(f"Material: {material}")
        if dims:     lines.append(f"Dimensions: {dims}")
        if brand:    lines.append(f"Brand: {brand}")
        if desc:     lines.append(desc)
        maxw = (col_price - 20) - col_desc
        wrapped = []
        for t in lines:
            wrapped.extend(wrap_to_width(t, maxw))

        if wrapped:
            draw_text(page, col_desc, y, wrapped[0])
        draw_right(page, col_price, y, price_str)
        y += line
        for t in wrapped[1:]:
            draw_text(page, col_desc, y, t); y += line

        # Row separator
        y += 6
        page.draw_rect(fitz.Rect(LM, y - 6, W - RM, y - 5), color=(0.85, 0.85, 0.85), fill=(0.85, 0.85, 0.85))
        y += 4

        if isinstance(price, (int, float)):
            net += float(price)

        if y > H - BM - 200:
            page = doc.new_page(width=W, height=H)
            y = TM

    # Totals
    y += line * 1.5
    vat = net * vat_rate
    total = net + vat
    draw_right(page, col_price, y, f"Net price: {fmt_price(net if net else None, sym)}"); y += line * 1.2
    draw_right(page, col_price, y, f"VAT (19%): {fmt_price(vat if net else None, sym)}"); y += line * 1.2
    draw_right(page, col_price, y, f"Total cost of the order: {fmt_price(total if net else None, sym)}"); y += line * 1.5

    # Terms
    draw_text(page, LM, y, "Terms:", fs); y += line
    for t in [
        "- Offer valid for 8 weeks",
        "- 40% advance payment due within 8 days",
        "- Balance due in 8 weeks",
    ]:
        draw_text(page, LM, y, t, fs); y += line

    y += line
    draw_text(page, LM, y, "Best regards,", fs); y += line
    draw_text(page, LM, y, "XYZ", fs); y += line

    doc.save(filename)
    doc.close()
    print(f"✅ PDF offer saved to: {filename}")


# ----------------------------- #
# Main
# ----------------------------- #
# ----------------------------- #
# Main
# ----------------------------- #
if __name__ == "__main__":
    customer_name = input("Enter customer name: ")
    customer_no = input("Enter customer number: ")
    user_query = input("Describe your furniture need: ")

    # Load + index
    chunked_docs, full_docs = load_offer_documents("offers")
    faiss_store, bm25_retriever = create_vectorstores(chunked_docs)

    # Match
    match_type, matched_docs = match_product(
        user_query, faiss_store, bm25_retriever,
        all_chunks=chunked_docs, full_docs=full_docs
    )

    if match_type == "none" or not matched_docs:
        print("❌ Sorry, we couldn’t find a matching product for that request.")
        raise SystemExit(0)

    if match_type == "similar":
        print("🤝 No exact match found. Closest items:")
        for i, d in enumerate(matched_docs, 1):
            print(f"\n[{i}] {d.metadata.get('source_file','?')}\n{d.page_content[:300]}...")
        confirm = input("\nProceed with a similar item? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Cancelled. No offer generated.")
            raise SystemExit(0)

    # Build context for extraction
    context = "\n\n---\n\n".join([d.page_content[:1500] for d in matched_docs])

    # Extract + dedupe
    items = extract_items(user_query, context, max_items=3)
    items = dedupe_items_by_name(items)

    # Ensure output folder exists
    output_dir = "generated_offers"
    os.makedirs(output_dir, exist_ok=True)

    # Generate PDF inside folder
    offer_id = "OFFER_" + datetime.now().strftime("%Y%m%d") + "_" + str(uuid.uuid4())[:4].upper()
    output_file = os.path.join(output_dir, f"{offer_id}.pdf")

    build_and_render_offer_pdf(
        offer_id=offer_id,
        customer_name=customer_name,
        customer_no=customer_no,
        items=items,
        vat_rate=0.19,
        default_currency="EUR",
        filename=output_file,
    )
    print(f"🎉 Offer successfully generated: {output_file}")

