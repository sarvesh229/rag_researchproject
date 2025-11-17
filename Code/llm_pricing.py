# -- coding: utf-8 --
"""
Offer generator (RAG + Excel-priced BOM with LLM dimension-based calculations)

Pipeline:
1) Parse ALL items from the user's free-text note (name/category/qty/color/dimensions/materials) via LLM.
2) For each item:
   a) Try to retrieve a matching catalog item from embedded PDFs and extract a priced line.
   b) If catalog fails, compute price from Excel using the LLM:
        - LLM uses dimensions_cm + Excel rows (unit, density_kg_per_m3, thickness_mm) to compute quantity.
        - Applies WASTAGE_FACTOR and CATEGORY_MIN_USD / GENERIC_MIN_USD.
        - Returns unit_price_est and price per item.
      If LLM gives 0/invalid, deterministic fallback pricing is applied from material map.
   c) If both fail, decline.
3) Build a PDF with Description + Dimensions and "Pricing: $X × Qty N = $Y" per item.

Hard rules:
- NEVER output $0.00: if LLM returns 0/invalid, deterministic fallback pricing is applied.
- Dimensions (W/D/H or Ø; H range) are guaranteed to print when present in note or BOM.
"""

import os, re, json, uuid, random
import fitz
import requests
from datetime import datetime
from dotenv import load_dotenv
import difflib
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DEBUG = True
COST_BOOK_PATH = "cost_book.xlsx"
EXCEL_SHEET = None  # auto-detect first sheet

# ---------- Pricing realism knobs ----------
WASTAGE_FACTOR = 1.25
CATEGORY_MIN_USD = {
    "table": 80.0, "desk": 100.0, "futon": 140.0, "sofa": 160.0,
    "shelf": 40.0, "chair": 50.0, "bed": 160.0, "nightstand": 50.0
}
GENERIC_MIN_USD = 50.0

# ---------- Decline messages ----------
DECLINE_ITEM_MSG = (
    "Sorry, we currently don’t have the product, but will notify you within 1 week if we can deliver it to you"
)
DECLINE_MSG = (
    "We currently are not able to fulllfill your requirenment but will contact you in 1 week "
    "to let you know if the similar item can be deilbverd to you"
)

# ----------------------------- #
# Together.ai Chat Completion
# ----------------------------- #
def together_chat(messages, model="mistralai/Mistral-7B-Instruct-v0.3", temperature=0.0, max_tokens=1200):
    load_dotenv()
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY not set in environment/.env")
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    if r.status_code != 200:
        raise Exception(f"API Error: {r.status_code} - {r.text}")
    return r.json()["choices"][0]["message"]["content"]

# ----------------------------- #
# Small numeric helper
# ----------------------------- #
def _to_float(x) -> Optional[float]:
    try:
        v = float(x)
        if not (v == v):
            return None
        if v in (float("inf"), float("-inf")):
            return None
        return v
    except Exception:
        return None

# ----------------------------- #
# JSON tolerance helpers
# ----------------------------- #
_TRAILING_COMMAS = re.compile(r",\s*([}\]])")
_SINGLE_QUOTE = re.compile(r"(?<=[:\s\[,])'([^']*)'(?=\s*[,}\]])")
def extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    t = _TRAILING_COMMAS.sub(r"\1", t[start : end + 1])
    t = _SINGLE_QUOTE.sub(r'"\1"', t)
    try:
        return json.loads(t)
    except Exception:
        return None

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
        raw_docs = PyMuPDFLoader(path).load()
        base = os.path.splitext(file)[0].lower()
        category_hint = base.replace("-", " ").replace("_", " ")
        first_token = category_hint.split()[0] if category_hint else ""
        whole = raw_docs[0].copy()
        whole.page_content = "\n".join(d.page_content for d in raw_docs)
        whole.metadata = (whole.metadata or {}) | {"source_file": file, "category": first_token}
        full_docs.append(whole)
        prepared = []
        for d in raw_docs:
            d.metadata = (d.metadata or {}) | {"source_file": file, "category": first_token}
            d.page_content = f"TITLE: {base}\nCONTENT:\n{d.page_content}"
            prepared.append(d)
        chunked_docs.extend(splitter.split_documents(prepared))
    if not chunked_docs:
        raise RuntimeError(f"No PDFs found in '{folder_path}'.")
    print(f"✅ Loaded {len(chunked_docs)} chunks from {len(full_docs)} PDFs.")
    return chunked_docs, full_docs

# ----------------------------- #
# Vector stores
# ----------------------------- #
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
def create_vectorstores(docs, faiss_path="faiss_index"):
    embedding = HuggingFaceEmbeddings(model_name=EMB_MODEL, encode_kwargs={"normalize_embeddings": True})
    vs = FAISS.from_documents(docs, embedding)
    vs.save_local(faiss_path)
    bm25 = BM25Retriever.from_documents(docs)
    return vs, bm25

# ----------------------------- #
# Intent utils / filters / search
# ----------------------------- #
FURNITURE_TYPES = [
    "bed","sofa","couch","chair","table","desk","wardrobe","cabinet","shelf",
    "bookshelf","dresser","nightstand","stool","bench","futon","recliner","mirror","lamp"
]
COLORS = [
    "black","white","brown","oak","walnut","grey","gray","beige","natural","espresso",
    "red","blue","copper","dark oak","dark walnut"
]
MATERIALS = [
    "wood","wooden","engineered wood","bamboo","rattan","teak","metal","steel","iron",
    "plywood","leather","fabric","plastic","glass","oak","pine","mdf","particleboard",
    "foam","glass fiber","fiberglass"
]
FOLD_SYNS = ("foldable","folding","collapsible","portable","fold-away","fold away","fold up")
EXCLUDE_KEYWORDS = [
    "connector","connectors","cover","covers","replacement","tool","clip","bolt","leg","kit",
    "spare","part","accessory","accessories","fastener","screw","bracket","joint","hardware"
]

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
        parts += ["table", "tables"] + list(FOLD_SYNS)
    if "light" in ql:
        parts += ["lightweight", "light weight", "light-weight", "portable"]
    return " ".join(parts)

def looks_like_main_furniture(text: str, intent: dict) -> bool:
    low = text.lower()
    t = intent.get("type")
    if t and t not in low:
        return False
    if any(bad in low for bad in EXCLUDE_KEYWORDS):
        return False
    good = {
        "chair","sofa","couch","table","desk","bench","stool","bed","wardrobe",
        "futon","recliner","shelf","nightstand","mirror","lamp"
    }
    return any(w in low for w in good) if t else True

def hybrid_search(query, faiss_store, bm25_retriever, k=18):
    dense = faiss_store.similarity_search_with_score(query, k=k * 2)
    sparse = bm25_retriever.get_relevant_documents(query)[: k * 2]
    scores, pool = {}, {}
    for i, (d, _) in enumerate(dense):
        pool[id(d)] = d
        scores[id(d)] = scores.get(id(d), 0.0) + 1.0 / (1 + i)
    for i, d in enumerate(sparse):
        pool[id(d)] = d
        scores[id(d)] = scores.get(id(d), 0.0) + 1.0 / (1 + i)
    ranked = sorted(pool.values(), key=lambda doc: scores[id(doc)], reverse=True)
    return ranked[:k]

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

# ----------------------------- #
# Catalog extraction (LLM + regex fallback)
# ----------------------------- #
JSON_SYSTEM = "You extract product line items from catalog text. Return STRICT JSON ONLY. No markdown, no comments."
PRICE_RE = re.compile(r'(?P<cur>€|EUR|\$|USD)\s*(?P<num>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)')
BAD_PREFIXES = ("material","dimensions","brand","pos","qty","description","price","net","vat","total")

def extract_items_via_llm(user_query, context, max_items=1):
    schema_hint = {
        "items": [
            {
                "pos": 1,
                "qty": 1,
                "unit": "pc",
                "name": "Product name",
                "material": "Wood",
                "dimensions": "200x160x40 cm",
                "brand": "Brand",
                "description": "Short line.",
                "price": 999.99,
                "currency": "USD",
            }
        ]
    }
    prompt = (
        "USER REQUEST:\n" + user_query + "\n\nPRODUCT CANDIDATES:\n" + context +
        "\n\nSelect up to " + str(max_items) +
        " items. Merge multi-line titles, extract brand/dimensions, numeric price if present (else null). "
        "Return strictly this JSON:\n" + json.dumps(schema_hint, ensure_ascii=False)
    )
    out = together_chat(
        [{"role": "system", "content": JSON_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=700,
    ).strip()
    out = re.sub(r"^```(?:json)?\s*|\s*```$", "", out, flags=re.I | re.S)
    try:
        data = json.loads(out)
        assert isinstance(data.get("items", []), list)
        return data
    except Exception:
        if DEBUG:
            print("LLM JSON parse failed for catalog:\n", out[:300], "...")
        return {"items": []}

def looks_like_title_line(s: str) -> bool:
    s = s.strip()
    if not s or re.match(r"^\s*(€|\$|usd|eur)\b", s.lower()):
        return False
    return 8 <= len(s) <= 200 and len(re.findall(r"[A-Za-z]", s)) >= 6

def _title_score(line: str, intent_type: Optional[str]) -> int:
    s = line.strip()
    if len(s) < 6 or len(s) > 120 or re.search(r"^\s*(€|\$|\d)", s):
        return -5
    score = 1
    if intent_type and intent_type in s.lower():
        score += 4
    score += min(sum(1 for w in re.findall(r"[A-Za-z]+", s) if w[0].isupper()), 4)
    if "," in s or " - " in s:
        score += 1
    return score

def extract_items_rule_based(context: str, intent: dict, max_items=1):
    lines = [l.strip() for l in context.splitlines() if l.strip()]
    items = []
    for i, line in enumerate(lines):
        m = PRICE_RE.search(line)
        if not m:
            continue
        window_idx = range(max(0, i - 8), min(len(lines), i + 9))
        best_k, best_sc = None, -999
        for k in window_idx:
            cand = lines[k]
            sc = _title_score(cand, intent.get("type"))
            if sc > best_sc and looks_like_title_line(cand):
                best_sc, best_k = sc, k
        if best_k is None:
            best_k = i
        title = lines[best_k]
        amount = None
        try:
            amount = float(re.sub(r"[^\d.]", "", m.group("num").replace(",", "")))
        except Exception:
            pass
        cur = "USD" if m.group("cur") in ("$", "USD") else "EUR"
        items.append(
            {
                "pos": len(items) + 1,
                "qty": 1,
                "unit": "pc",
                "name": title or "Item",
                "material": None,
                "dimensions": None,
                "brand": None,
                "description": "",
                "price": amount,
                "currency": cur,
            }
        )
        if len(items) >= max_items:
            break
    return items

def catalog_best_item_for_spec(
    spec: Dict[str, Any],
    faiss_store,
    bm25_retriever,
    all_chunks,
    full_docs,
) -> Optional[Dict[str, Any]]:
    """
    Build a query from the BOM spec; search catalog; try to extract 1 priced line.
    Returns a line-item dict or None.
    """
    name = spec.get("name") or spec.get("category") or ""
    color = spec.get("color") or ""
    mats = " ".join(m.get("name", "") for m in spec.get("materials", []))
    base_query = f"{name} {color} {mats}".strip()
    intent = parse_intent(base_query or name)

    cands = hybrid_search(base_query, faiss_store, bm25_retriever, k=12)
    cands = metadata_filter(cands, intent)
    if not cands:
        return None

    d = cands[0]
    context = d.page_content[:1500]
    ext = extract_items_via_llm(name, context, max_items=1)
    items = [it for it in ext.get("items", []) if it.get("name")]
    if not items:
        rb = extract_items_rule_based(context, intent, max_items=1)
        items = rb or []

    if not items:
        return None

    it = items[0]
    qty = int(spec.get("qty") or 1)
    if it.get("price") in (None, "", 0, 0.0):
        return None

    dims_txt = None
    if it.get("dimensions"):
        dims_txt = it["dimensions"]
    else:
        dims = spec.get("dimensions_cm", {}) or {}
        parts = []
        if "diameter" in dims:
            p = [f'{dims["diameter"]}Ø cm']
            if "h_min" in dims and "h_max" in dims:
                p.append(f'{dims["h_min"]}-{dims["h_max"]}H cm')
            elif "h" in dims:
                p.append(f'{dims["h"]}H cm')
            dims_txt = "; ".join(p)
        else:
            if "w" in dims:
                parts.append(f'{dims["w"]}W')
            if "d" in dims:
                parts.append(f'{dims["d"]}D')
            if "h" in dims:
                parts.append(f'{dims["h"]}H')
            if parts:
                dims_txt = " x ".join(parts) + " (cm)"

    materials_list = " / ".join(sorted({m["name"] for m in spec.get("materials", []) if m.get("name")}))
    line_total = float(it.get("price"))
    unit_est = line_total / max(1, qty)

    return {
        "pos": 0,
        "qty": qty,
        "unit": "pc",
        "name": (it.get("name") or name or "Item").strip().title(),
        "material": materials_list or it.get("material"),
        "dimensions": dims_txt,
        "brand": it.get("brand"),
        "description": it.get("description") or (spec.get("color") or "Custom build per customer specification."),
        "price": round(line_total, 2),
        "unit_price_est": round(unit_est, 2),
        "currency": it.get("currency") or "USD",
    }

# ----------------------------- #
# Excel pricing (baseline + for LLM prompt)
# ----------------------------- #
REQUIRED_COLS = {"material_name", "unit", "unit_price_usd"}  # others optional

def autodetect_sheet(path: str) -> str:
    xl = pd.ExcelFile(path)
    name = EXCEL_SHEET or (xl.sheet_names[0] if xl.sheet_names else "Sheet1")
    print(f"📄 Loaded sheet: {name}")
    return name

def load_material_costs_xlsx(path=COST_BOOK_PATH, sheet: Optional[str] = None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cost book not found at '{path}'.")
    if sheet is None:
        sheet = autodetect_sheet(path)
    df = pd.read_excel(path, sheet_name=sheet).fillna("")
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Cost sheet missing columns: {', '.join(sorted(missing))}")
    df["material_name_norm"] = df["material_name"].astype(str).str.strip().str.lower()
    df["unit_norm"] = df["unit"].astype(str).str.strip().str.lower()

    def _num(col):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.Series([float("nan")] * len(df))

    _num("unit_price_usd")
    _num("density_kg_per_m3")
    _num("thickness_mm")

    price_map = {
        f"{row['material_name_norm']}|{row['unit_norm']}": float(row["unit_price_usd"])
        for _, row in df.iterrows()
        if pd.notna(row["unit_price_usd"])
    }

    material_to_units: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        m = row["material_name_norm"]
        u = row["unit_norm"]
        p = row["unit_price_usd"]
        if pd.notna(p):
            material_to_units.setdefault(m, {})[u] = float(p)

    allowed_materials = sorted(df["material_name_norm"].unique().tolist())
    return price_map, material_to_units, allowed_materials, df

def price_from_bom(spec: Dict[str, Any], price_map: Dict[str, float], material_to_units: Dict[str, Dict[str, float]]) -> float:
    total = 0.0
    for m in spec.get("materials", []):
        name = (m.get("name") or "").strip().lower()
        unit = (m.get("unit") or "").strip().lower()
        qty = float(m.get("quantity") or 0.0)
        if not name or not unit or qty <= 0:
            continue
        price = price_map.get(f"{name}|{unit}") or material_to_units.get(name, {}).get(unit)
        if price is None:
            if DEBUG:
                print(f"⚠ Missing price for material: {name} ({unit})")
            continue
        total += qty * float(price)
    return round(total, 2)

def adjusted_unit_price(category: str, raw_unit_cost: float) -> float:
    base = raw_unit_cost * WASTAGE_FACTOR
    floor = CATEGORY_MIN_USD.get((category or "").lower(), GENERIC_MIN_USD)
    return max(base, floor)

# ----------------------------- #
# LLM-based pricing with full Excel rows and DIMENSIONS
# ----------------------------- #
def build_price_table_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    for _, row in df.iterrows():
        rec = {}
        for c in df.columns:
            v = row[c]
            if isinstance(v, float) and (v != v):  # NaN
                v = ""
            rec[c] = v
        out.append(rec)
    return out

def llm_compute_prices_for_bom(
    items_bom: List[Dict[str, Any]],
    price_table_rows: List[Dict[str, Any]],
    wastage_factor: float,
    category_min_usd: Dict[str, float],
    generic_min_usd: float,
) -> Optional[List[Dict[str, Any]]]:
    """
    Ask the LLM to compute prices using:
    - dimensions_cm (cm)
    - Excel rows (unit, unit_price_usd, density_kg_per_m3, thickness_mm)
    - explicit geometric + unit conversion rules
    """

    system = (
        "You are a careful pricing calculator. "
        "Return ONLY a JSON object with an 'items' array. No prose, no markdown. "
        "Use ONLY the provided price_table_rows for unit prices. "
        "You MUST compute material quantity from dimensions_cm (in cm) and units."
    )

    rules = {
        "dimensions": [
            "All dimensions in dimensions_cm are in centimeters.",
            "If dimensions_cm has w and d: rectangular top area_m2 = (w/100) * (d/100).",
            "If dimensions_cm has diameter: radius_m = (diameter/100)/2; round top area_m2 = π * radius_m^2.",
            "If dimensions_cm has h: height_m = h/100.",
            "If dimensions_cm has h_min and h_max: height_m = ((h_min + h_max)/2)/100."
        ],
        "unit_conversions": [
            "If Excel unit == 'm2': quantity_in_unit = top surface area in m2 (round if available, else rectangular).",
            "If Excel unit == 'm3': quantity_in_unit = volume_m3.",
            "volume_m3 = area_m2 * height_m when possible.",
            "If volume_m3 is missing but thickness_mm is available and area_m2 is known: "
            "    thickness_m = thickness_mm / 1000; volume_m3 = area_m2 * thickness_m.",
            "If Excel unit == 'kg' and density_kg_per_m3 is available: quantity_in_unit = volume_m3 * density_kg_per_m3.",
            "If you cannot compute quantity_in_unit reliably, set raw_unit_cost = 0 (do not invent)."
        ],
        "material_matching": [
            "For each item, choose the most relevant material row(s) by matching material_name to "
            "item.materials[].name or item.category, case-insensitive.",
            "If multiple rows match, prefer the most specific (e.g., more detailed material)."
        ],
        "pricing": [
            "For each item: raw_unit_cost = sum(quantity_in_unit * unit_price_usd for all chosen materials).",
            "unit_price_est = max(raw_unit_cost * WASTAGE_FACTOR, category_min_usd.get(category, generic_min_usd)).",
            "line_total = unit_price_est * qty.",
            "Round all monetary values to 2 decimals."
        ],
        "output_schema": {
            "items": [
                {
                    "name": "Item name",
                    "category": "bed",
                    "qty": 1,
                    "unit_price_est": 0.0,
                    "price": 0.0,
                    "debug": {
                        "area_m2": 0.0,
                        "volume_m3": 0.0,
                        "quantity_in_unit": 0.0,
                        "raw_unit_cost": 0.0,
                        "used_unit": "m2"
                    }
                }
            ]
        },
        "constraints": [
            "Never invent unit prices. Use only unit_price_usd from price_table_rows.",
            "Never output negative prices.",
            "If you cannot compute or all quantities are zero, set unit_price_est = 0 and price = 0 (do NOT guess)."
        ]
    }

    payload = {
        "items_bom": items_bom,
        "price_table_rows": price_table_rows,
        "WASTAGE_FACTOR": wastage_factor,
        "category_min_usd": category_min_usd,
        "generic_min_usd": generic_min_usd,
        "rules": rules,
    }

    def _ask(minimal=False):
        msgs = [
            {
                "role": "system",
                "content": "Return ONLY JSON. No markdown, no comments." if minimal else system,
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        out = together_chat(msgs, temperature=0.0, max_tokens=2200).strip()
        return extract_json_object(out)

    obj = _ask(minimal=False) or _ask(minimal=True)
    if not obj or not isinstance(obj.get("items"), list):
        if DEBUG:
            print("⚠ LLM pricing parse failed.")
        return None
    return obj["items"]

def build_priced_items_from_bom_llm(
    items_bom: List[Dict[str, Any]],
    price_map: Dict[str, float],
    material_to_units: Dict[str, Dict[str, float]],
    df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """
    Build priced items using:
    - LLM dimension-based pricing from Excel rows
    - Deterministic fallback if LLM gives 0/invalid
    """
    price_table = build_price_table_from_df(df if df is not None else pd.DataFrame())
    llm_prices = llm_compute_prices_for_bom(
        items_bom=items_bom,
        price_table_rows=price_table,
        wastage_factor=WASTAGE_FACTOR,
        category_min_usd=CATEGORY_MIN_USD,
        generic_min_usd=GENERIC_MIN_USD,
    )

    out: List[Dict[str, Any]] = []

    def dim_text(spec: Dict[str, Any]) -> Optional[str]:
        dims = spec.get("dimensions_cm", {}) or {}
        if not dims:
            return None
        if "diameter" in dims:
            parts = [f'{dims["diameter"]}Ø cm']
            if "h_min" in dims and "h_max" in dims:
                parts.append(f'{dims["h_min"]}-{dims["h_max"]}H cm')
            elif "h" in dims:
                parts.append(f'{dims["h"]}H cm')
            return "; ".join(parts)
        parts = []
        if "w" in dims:
            parts.append(f'{dims["w"]}W')
        if "d" in dims:
            parts.append(f'{dims["d"]}D')
        if "h" in dims:
            parts.append(f'{dims["h"]}H')
        return " x ".join(parts) + " (cm)" if parts else None

    idx: Dict[str, Dict[str, Any]] = {}
    if llm_prices:
        for it in llm_prices:
            key = (it.get("name") or it.get("category") or "").strip().lower()
            if key:
                idx[key] = it

    for spec in items_bom:
        key = (spec.get("name") or spec.get("category") or "").strip()
        k = key.lower()
        qty = int(spec.get("qty") or 1)
        if qty < 1:
            qty = 1
        materials_list = " / ".join(sorted({m["name"] for m in spec.get("materials", []) if m.get("name")}))

        unit_est = line_total = None
        if idx:
            hit = idx.get(k)
            if hit:
                unit_est = _to_float(hit.get("unit_price_est"))
                line_total = _to_float(hit.get("price"))

        # Fallback if LLM gave invalid/0
        if unit_est is None or unit_est <= 0 or line_total is None or line_total <= 0:
            if DEBUG:
                print(f"⚠ LLM price invalid for '{key}', falling back to material map pricing.")
            unit_raw = price_from_bom(spec, price_map, material_to_units)
            unit_adj = adjusted_unit_price(spec.get("category", ""), unit_raw)
            unit_est = round(unit_adj, 2)
            line_total = round(unit_adj * qty, 2)

        out.append(
            {
                "pos": 0,
                "qty": qty,
                "unit": "pc",
                "name": key.strip().title() or "Custom Item",
                "material": materials_list or None,
                "dimensions": dim_text(spec),
                "brand": None,
                "description": (spec.get("color") or "Custom build per customer specification."),
                "price": float(line_total),
                "unit_price_est": float(unit_est),
                "currency": "USD",
            }
        )

    return out

def build_priced_items_from_bom(items_bom: List[Dict[str, Any]], price_map, material_to_units, df=None) -> List[Dict[str, Any]]:
    return build_priced_items_from_bom_llm(items_bom, price_map, material_to_units, df=df)

# ----------------------------- #
# Multi-item BOM from user notes
# ----------------------------- #
def build_multi_bom_prompt(user_text: str, allowed_materials: List[str]) -> str:
    mats = "\n".join(f"- {m}" for m in allowed_materials[:300])
    return f"""
SYSTEM: Extract ALL distinct furniture items from the note and output JSON with an "items" array.

DIMENSIONS & UNITS RULES:
- dimensions_cm MUST always be in centimeters (cm).
- If the note uses mm, convert to cm (e.g., 830 mm = 83 cm).
- Allowed keys in dimensions_cm: w, d, h, diameter, h_min, h_max.

MATERIAL RULES:
- Use ONLY materials from the allowed list below.
- You may choose up to 3 materials per item.
- Do NOT guess material quantities precisely; a rough quantity is allowed, but final pricing will be based on dimensions.
- Recommended default units:
  - wood/mdf/pine/foam/plastic/fiberglass in m³
  - metal/steel/iron in kg
  - leather/fabric/glass/marble in m²

ITEM SCHEMA:
- Each item MUST have:
  - name
  - category (bed, sofa, chair, table, desk, futon, nightstand, mirror, etc.)
  - qty
  - color
  - dimensions_cm (with correct cm conversion)
  - materials: a list of {{name, unit, quantity}} (approximate quantity is OK).

CONSTRAINTS:
- Respect counts in the note (e.g., "set of two chairs" = qty 2).
- Do NOT invent extra items.
- Return JSON ONLY, no prose.

ALLOWED_MATERIALS:
{mats}

USER NOTE:
{user_text}

OUTPUT EXAMPLE:
{{
  "items": [
    {{
      "name": "Multi-Layer Bed",
      "category": "bed",
      "qty": 1,
      "color": "dark walnut finish",
      "dimensions_cm": {{"w": 151, "d": 39, "h": 98}},
      "materials": [{{"name":"plastic","unit":"m³","quantity":0.10}}]
    }},
    {{
      "name": "Compact Sofa",
      "category": "sofa",
      "qty": 1,
      "color": "beige / matte black frame",
      "dimensions_cm": {{"w": 82, "d": 78, "h": 29}},
      "materials": [
        {{"name":"glass fiber","unit":"m³","quantity":0.06}},
        {{"name":"foam","unit":"m³","quantity":0.08}},
        {{"name":"fabric","unit":"m²","quantity":6.0}}
      ]
    }},
    {{
      "name": "Wooden Nightstand",
      "category": "nightstand",
      "qty": 1,
      "color": "walnut tone",
      "dimensions_cm": {{"w": 42, "h": 41}},
      "materials": [{{"name":"oak","unit":"m³","quantity":0.02}}]
    }}
  ]
}}
"""

def get_multi_bom_from_llm(user_text: str, allowed_materials: List[str]) -> List[Dict[str, Any]]:
    prompt = build_multi_bom_prompt(user_text, allowed_materials)
    for _ in range(2):
        out = together_chat(
            [
                {"role": "system", "content": "Return ONLY the JSON object with an 'items' array; no prose."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1600,
        ).strip()
        obj = extract_json_object(out)
        if obj and isinstance(obj.get("items"), list) and obj["items"]:
            return obj["items"]
        if DEBUG:
            print("Multi-BOM parse failed. Raw:", out[:400], "...")
    return []

# ----------------------------- #
# PDF builder (with Dimensions column)
# ----------------------------- #
def currency_symbol(cur: str) -> str:
    return {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}.get((cur or "USD").upper(), "$")

def build_and_render_offer_pdf(
    offer_id,
    customer_name,
    customer_no,
    items,
    vat_rate=0.0,
    default_currency="USD",
    filename="offer.pdf",
):
    import fitz
    from datetime import datetime

    W, H = 595, 842  # A4 portrait
    LM, TM, RM, BM = 36, 36, 36, 36
    y = TM
    line = 16
    fs_h1, fs_h2, fs, fs_small = 18, 12, 11, 10

    sym = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}.get(
        (items[0].get("currency") if items else default_currency).upper(), "$"
    )
    today = datetime.now().strftime("%d.%m.%Y")

    col_pos = LM
    col_qty = col_pos + 40
    col_desc = col_qty + 44
    col_dim = col_desc + 240
    col_price = W - RM

    def draw_text(page, x, y, text, size=fs):
        page.insert_text(fitz.Point(x, y), text, fontsize=size, color=(0, 0, 0))

    def draw_right(page, x_right, y, text, size=fs):
        rect = fitz.Rect(LM, y - 12, x_right, y + 20)
        try:
            page.insert_textbox(rect, text, fontsize=size, align=fitz.TEXT_ALIGN_RIGHT)
        except TypeError:
            page.insert_text(fitz.Point(x_right - len(text) * 6, y), text, fontsize=size, color=(0, 0, 0))

    def wrap_to_width(text, max_width_pt):
        if not text:
            return []
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

    def fmt_price(amount):
        if amount is None:
            return "TBD"
        return f"{sym}{amount:,.2f}"

    doc = fitz.open()
    page = doc.new_page(width=W, height=H)

    draw_text(page, LM, y, f"Offer ID: {offer_id}", fs_h1)
    y += line * 1.6
    draw_text(page, LM, y, f"Customer: {customer_name}", fs_h2)
    y += line
    draw_text(page, LM, y, f"Date: {today}", fs_h2)
    y += line * 1.4
    draw_text(page, LM, y, "We are pleased to present you with an offer for the following items:", fs)
    y += line * 1.3

    page.draw_rect(fitz.Rect(LM, y - 10, W - RM, y - 9), color=(0, 0, 0), fill=(0, 0, 0))
    draw_text(page, col_pos, y, "Pos", fs_small)
    draw_text(page, col_qty, y, "Qty", fs_small)
    draw_text(page, col_desc, y, "Description", fs_small)
    draw_text(page, col_dim, y, "Dimensions", fs_small)
    draw_right(page, col_price, y, "Price", fs_small)
    y += line

    net = 0.0
    for idx, it in enumerate(items, 1):
        pos = f"{idx:03}"
        qty = int(it.get("qty", 1))
        unit = it.get("unit", "pc")
        name = (it.get("name") or "Item").strip()
        material = it.get("material")
        dims_txt = it.get("dimensions") or ""
        brand = it.get("brand")
        desc = it.get("description") or ""
        line_total = it.get("price")
        unit_est = it.get("unit_price_est")

        desc_lines = [name]
        if material:
            desc_lines.append(f"Material: {material}")
        if brand:
            desc_lines.append(f"Brand: {brand}")
        if desc:
            desc_lines.append(desc)
        if unit_est is not None:
            desc_lines.append(f"Pricing: {fmt_price(unit_est)} × Qty {qty} = {fmt_price(line_total)}")

        wrapped_desc = []
        for t in desc_lines:
            wrapped_desc.extend(wrap_to_width(t, (col_dim - 10) - col_desc))

        wrapped_dim = wrap_to_width(dims_txt, (col_price - 20) - col_dim) if dims_txt else []

        row_lines = max(1, max(len(wrapped_desc), len(wrapped_dim)))

        draw_text(page, col_pos, y, pos)
        draw_text(page, col_qty, y, str(qty))
        draw_text(page, col_qty, y + line * 0.9, unit, fs_small)

        if wrapped_desc:
            draw_text(page, col_desc, y, wrapped_desc[0])
        else:
            draw_text(page, col_desc, y, name)

        if wrapped_dim:
            draw_text(page, col_dim, y, wrapped_dim[0])

        draw_right(page, col_price, y, fmt_price(line_total))

        for i in range(1, row_lines):
            if i < len(wrapped_desc):
                draw_text(page, col_desc, y + i * line, wrapped_desc[i])
            if i < len(wrapped_dim):
                draw_text(page, col_dim, y + i * line, wrapped_dim[i])

        y += row_lines * line + 6
        page.draw_rect(
            fitz.Rect(LM, y - 6, W - RM, y - 5),
            color=(0.85, 0.85, 0.85),
            fill=(0.85, 0.85, 0.85),
        )
        y += 4

        if isinstance(line_total, (int, float)):
            net += float(line_total)

        if y > H - BM - 200:
            page = doc.new_page(width=W, height=H)
            y = TM

    y += line * 1.5
    vat = net * vat_rate
    total = net + vat
    draw_right(page, col_price, y, f"Net price: {fmt_price(net if net else None)}")
    y += line * 1.2
    draw_right(page, col_price, y, f"VAT ({int(vat_rate * 100)}%): {fmt_price(vat if net else None)}")
    y += line * 1.2
    draw_right(page, col_price, y, f"Total cost of the order: {fmt_price(total if net else None)}")
    y += line * 1.5

    for t in [
        "- Offer valid for 8 weeks",
        "- 40% advance payment due within 8 days",
        "- Balance due in 8 weeks",
    ]:
        draw_text(page, LM, y, t, fs)
        y += line

    y += line
    draw_text(page, LM, y, "Best regards,", fs)
    y += line
    draw_text(page, LM, y, "XYZ", fs)
    y += line

    doc.save(filename)
    doc.close()
    print(f"✅ PDF offer saved to: {filename}")

# ----------------------------- #
# NOTE → DIMENSIONS HELPERS (for display override)
# ----------------------------- #
DIM_MM = r"(\d{2,4})\s*mm"

def mm_to_cm(mm: str) -> Optional[float]:
    try:
        return round(float(mm) / 10.0)  # to nearest cm
    except Exception:
        return None

def extract_dimensions_from_note(note: str) -> dict:
    """
    Return canonical dimension strings per item category from a free-text note.
    Keys: 'bed', 'sofa', 'nightstand', 'desk', 'table'
    Values: '80W x 60D x 90H (cm)', '60Ø; 98-123H (cm)', etc.
    """
    s = " ".join(note.lower().split())
    dims = {}

    # Bed
    m = re.search(
        r"bed[^.]*?width[^0-9]*"
        + DIM_MM
        + r"[^.]*?height[^0-9]*"
        + DIM_MM
        + r"[^.]*?depth[^0-9]*"
        + DIM_MM,
        s,
    )
    if m:
        w = mm_to_cm(m.group(1))
        h = mm_to_cm(m.group(2))
        d = mm_to_cm(m.group(3))
        parts = []
        if w is not None:
            parts.append(f"{w}W")
        if d is not None:
            parts.append(f"{d}D")
        if h is not None:
            parts.append(f"{h}H")
        if parts:
            dims["bed"] = " x ".join(parts) + " (cm)"

    # Sofa
    m = re.search(
        r"sofa[^.]*?(?:wide|width)[^0-9]*"
        + DIM_MM
        + r"[^.]*?(?:deep|depth)[^0-9]*"
        + DIM_MM
        + r"[^.]*?(?:high|height)[^0-9]*"
        + DIM_MM,
        s,
    )
    if m:
        w = mm_to_cm(m.group(1))
        d = mm_to_cm(m.group(2))
        h = mm_to_cm(m.group(3))
        parts = []
        if w is not None:
            parts.append(f"{w}W")
        if d is not None:
            parts.append(f"{d}D")
        if h is not None:
            parts.append(f"{h}H")
        if parts:
            dims["sofa"] = " x ".join(parts) + " (cm)"

    # Nightstand
    m = re.search(
        r"nightstand[^.]*?(?:wide|width)[^0-9]*"
        + DIM_MM
        + r"[^.]*?(?:high|height)[^0-9]*"
        + DIM_MM,
        s,
    )
    if m:
        w = mm_to_cm(m.group(1))
        h = mm_to_cm(m.group(2))
        parts = []
        if w is not None:
            parts.append(f"{w}W")
        if h is not None:
            parts.append(f"{h}H")
        if parts:
            dims["nightstand"] = " x ".join(parts) + " (cm)"

    # Desk (height only)
    m = re.search(r"desk[^.]*?" + DIM_MM + r"[^.]*?(?:tall|high|height)", s)
    if m:
        h = mm_to_cm(m.group(1))
        if h is not None:
            dims["desk"] = f"{h}H (cm)"

    # Round table
    m_d = re.search(r"(?:round\s+)?table[^.]*?" + DIM_MM + r"[^.]*?(?:diameter|ø)", s)
    m_h = re.search(
        r"(?:round\s+)?table[^.]*?(\d{2,3})\s*cm\s*(?:to|-|–|and)\s*(\d{2,3})\s*cm", s
    )
    parts = []
    if m_d:
        d_cm = mm_to_cm(m_d.group(1))
        if d_cm is not None:
            parts.append(f"{d_cm}Ø")
    if m_h:
        parts.append(f"{m_h.group(1)}-{m_h.group(2)}H")
    if parts:
        dims["table"] = "; ".join(parts) + " (cm)"

    return dims

def inject_dimensions_into_items(items: list, note_dims: dict) -> list:
    if not items or not note_dims:
        return items

    def pick_key(name: str) -> Optional[str]:
        n = (name or "").lower()
        if "futon" in n:
            return "futon"
        if "sofa" in n or "sectional" in n or "couch" in n:
            return "sofa"
        if "desk" in n:
            return "desk"
        if "table" in n:
            return "table"
        if "nightstand" in n:
            return "nightstand"
        if "bed" in n:
            return "bed"
        return None

    for it in items:
        key = pick_key(it.get("name", ""))
        if key and note_dims.get(key):
            # Only override DISPLAY dimensions; pricing uses dimensions_cm inside BOM
            it["dimensions"] = note_dims[key]
    return items

# ----------------------------- #
# Customer name/number parsing
# ----------------------------- #
def parse_customer_from_note(note: str) -> Tuple[str, str]:
    m = re.search(r"\bcustomer\s+([A-Z][a-zA-Z\-]+)", note)
    name = (m.group(1).strip().title() if m else "Customer")
    number = str(random.randint(1000, 9999))
    return name, number

# ----------------------------- #
# MAIN
# ----------------------------- #
if __name__ == "__main__":
    user_query = input("Paste the furniture note / request: ").strip()

    customer_name, _ = parse_customer_from_note(user_query)

    if customer_name.lower() == "customer":
        customer_name = input("Please enter the customer's name: ").strip().title()
        if not customer_name:
            print("❌ Customer name is required to generate the offer.")
            raise SystemExit(0)

    # Load data
    price_map, material_to_units, allowed_materials, df_cost = load_material_costs_xlsx(
        COST_BOOK_PATH, EXCEL_SHEET
    )
    chunked_docs, full_docs = load_offer_documents("offers")
    faiss_store, bm25_retriever = create_vectorstores(chunked_docs)

    # 1) Parse ALL items from the note
    bom_items = get_multi_bom_from_llm(user_query, allowed_materials)
    if not bom_items:
        print(DECLINE_MSG)
        raise SystemExit(0)

    # 2) For each item: catalog → Excel (LLM + dims) → decline
    final_items: List[Dict[str, Any]] = []
    unavailable = False

    for spec in bom_items:
        # Try catalog
        cat_item = catalog_best_item_for_spec(
            spec, faiss_store, bm25_retriever, chunked_docs, full_docs
        )

        if cat_item:
            final_items.append(cat_item)
            continue

        # Else price from Excel via LLM (dimension-based) with deterministic fallback guard
        priced = build_priced_items_from_bom(
            [spec], price_map, material_to_units, df=df_cost
        )
        if priced and priced[0].get("price", 0) > 0:
            final_items.extend(priced)
            continue

        # Else mark unavailable and stop
        unavailable = True
        break

    if unavailable or not final_items:
        print(DECLINE_ITEM_MSG)
        raise SystemExit(0)

    # 3) Inject human-friendly dimensions parsed directly from note (for display only)
    note_dims = extract_dimensions_from_note(user_query)
    final_items = inject_dimensions_into_items(final_items, note_dims)

    # Assign positions
    for i, it in enumerate(final_items, 1):
        it["pos"] = i

    os.makedirs("generated_offers", exist_ok=True)
    offer_id = "OFFER_" + datetime.now().strftime("%Y%m%d") + "_" + str(uuid.uuid4())[:4].upper()
    output_file = os.path.join("generated_offers", f"{offer_id}.pdf")

    build_and_render_offer_pdf(
        offer_id=offer_id,
        customer_name=customer_name,
        customer_no="",
        items=final_items,
        vat_rate=0.00,
        default_currency="USD",
        filename=output_file,
    )
    print(f"🎉 Offer successfully generated: {output_file}")
