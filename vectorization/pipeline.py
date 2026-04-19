"""
Pipeline: row from Загрузка_2 → GPT-4o-mini (photo + text) → normalize → embed → pgvector top-10

Usage:
    python pipeline.py 112          # search single row
    python pipeline.py 1 50         # batch test rows 1-50, print accuracy
"""

import os
import sys
import base64
import psycopg2
import pandas as pd
import openpyxl
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

EXCEL_PATH = os.path.join(os.path.dirname(__file__), "../data_files/Загрузка 2.xlsx")
SHEET_NAME = "装车明细"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_db():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "customs_vectors"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
    )


# ---------------------------------------------------------------------------
# Load Excel once (pandas for data, openpyxl for images)
# ---------------------------------------------------------------------------

_df = None
_ws = None


def _load():
    global _df, _ws
    if _df is None:
        print("Loading Excel…", end=" ", flush=True)
        _df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=0)
        _df.columns = _df.columns.str.strip()
        wb = openpyxl.load_workbook(EXCEL_PATH, read_only=False, data_only=True)
        _ws = wb[SHEET_NAME]
        print("done.")
    return _df, _ws


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------

def _get_row_images(ws, pandas_row_idx: int) -> list[bytes]:
    """Return raw bytes for all images anchored to the given pandas row (0-indexed)."""
    # pandas row i  →  Excel row i+2 (1-indexed)  →  anchor._from.row = i+1 (0-indexed)
    anchor_row = pandas_row_idx + 1
    images = []
    for img in ws._images:
        if hasattr(img.anchor, "_from") and img.anchor._from.row == anchor_row:
            img.ref.seek(0)
            images.append(img.ref.read())
    return images


def _encode_images(raw_images: list[bytes]) -> list[dict]:
    """Build OpenAI vision content blocks from raw image bytes."""
    blocks = []
    for raw in raw_images:
        b64 = base64.b64encode(raw).decode()
        blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })
    return blocks


# ---------------------------------------------------------------------------
# Row data extraction
# ---------------------------------------------------------------------------

TEXT_COLS = [
    ("артикул",      "货号(артикул)"),
    ("наименование", "名称(наименование)"),
    ("состав",       "成分（состав)"),
    ("изготовитель", "изготовитель"),
    ("ширина_м",     "尺寸宽/（шилина（M)"),
    ("длина_м",      "长длина(M)"),
    ("длина_см",     "长(длина)"),
    ("ширина_см",    "宽(ширина)"),
    ("высота_см",    "高(высота)"),
]


def _row_text(df: pd.DataFrame, pandas_row_idx: int) -> dict:
    row = df.iloc[pandas_row_idx]
    data = {}
    for label, col in TEXT_COLS:
        if col in df.columns:
            val = row[col]
            if pd.notna(val) and str(val).strip() not in ("", "nan"):
                data[label] = str(val).strip()
    return data


def _ground_truth(df: pd.DataFrame, pandas_row_idx: int) -> str | None:
    row = df.iloc[pandas_row_idx]
    val = row.get("код")
    if pd.isna(val):
        return None
    code = str(val).split(".")[0].strip()
    return code.zfill(10) if code.isdigit() else code


# ---------------------------------------------------------------------------
# GPT calls
# ---------------------------------------------------------------------------

def _describe(text_fields: dict, image_blocks: list[dict]) -> str:
    """GPT-4o-mini: photo + text → customs declaration description (single call)."""
    field_str = "\n".join(f"  {k}: {v}" for k, v in text_fields.items())
    prompt = (
        "Ты таможенный эксперт. Опиши товар так, как он записывается в таможенной декларации. "
        "Укажи: точный тип изделия, материал изготовления, назначение, ключевые технические характеристики. "
        "Используй таможенную терминологию. Ответ — одно предложение на русском языке.\n\n"
        f"Данные товара:\n{field_str}"
    )
    content = [{"type": "text", "text": prompt}] + image_blocks
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        max_tokens=150,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def _rerank(query_description: str, candidates: list[tuple[str, str, float]]) -> tuple[str, str]:
    """GPT-4o-mini: pick the best HS code from top-10 candidates."""
    candidates_str = "\n".join(
        f"{i+1}. Код {code}: {desc[:120]}"
        for i, (code, desc, _) in enumerate(candidates)
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Товар: {query_description}\n\n"
                f"Кандидаты ТН ВЭД:\n{candidates_str}\n\n"
                "Выбери один наиболее подходящий код ТН ВЭД для этого товара. "
                "Ответь строго в формате:\nКОД: <10-значный код>\nПРИЧИНА: <одно предложение>"
            ),
        }],
        max_tokens=100,
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()
    code, reason = "", text
    for line in text.splitlines():
        if line.startswith("КОД:"):
            code = line.replace("КОД:", "").strip()
        elif line.startswith("ПРИЧИНА:"):
            reason = line.replace("ПРИЧИНА:", "").strip()
    return code, reason


def _embed(text: str) -> list[float]:
    resp = client.embeddings.create(input=text, model="text-embedding-3-small")
    return resp.data[0].embedding


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

def _search(embedding: list[float], top_k: int = 10) -> list[tuple[str, str, float]]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT code, description, (embedding <=> %s::vector) AS distance
        FROM tnved_codes
        ORDER BY distance ASC
        LIMIT %s
        """,
        (embedding, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows  # [(code, description, distance), ...]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_row(excel_row_number: int, verbose: bool = True) -> dict:
    """
    excel_row_number: 1-indexed visible row number (matches what Nikita shows).
    Returns dict with all intermediate results.
    """
    df, ws = _load()
    pandas_idx = excel_row_number - 2  # header is row 1, data starts at row 2

    if pandas_idx < 0 or pandas_idx >= len(df):
        print(f"Row {excel_row_number} out of range (1-indexed, data rows 2–{len(df)+1})")
        return {}

    text_fields = _row_text(df, pandas_idx)
    raw_images = _get_row_images(ws, pandas_idx)
    image_blocks = _encode_images(raw_images)
    real_code = _ground_truth(df, pandas_idx)

    if verbose:
        print(f"\n{'='*62}")
        print(f"Строка #{excel_row_number}  |  Реальный код: {real_code or '—'}")
        print("="*62)
        for k, v in text_fields.items():
            print(f"  {k}: {v}")
        print(f"\nФото: {len(raw_images)} шт → отправляю в gpt-4o-mini…")

    description = _describe(text_fields, image_blocks)
    embedding = _embed(description)
    results = _search(embedding)
    best_code, reason = _rerank(description, results)

    if verbose:
        print(f"Описание:   «{description}»")
        print("-" * 62)
        for i, (code, desc, dist) in enumerate(results, 1):
            marker = " ✓" if code == real_code else ""
            print(f"  #{i:2d}  {code}   {1-dist:.4f}{marker}")
            print(f"        {desc[:80]}")
        print(f"\n→ GPT выбрал: {best_code}  {'✓ ВЕРНО' if best_code == real_code else '✗ НЕВЕРНО (реальный: '+str(real_code)+')'}")
        print(f"  Причина: {reason}")
        print()

    top_codes = [r[0] for r in results]
    vector_rank = (top_codes.index(real_code) + 1) if real_code in top_codes else -1
    rerank_correct = best_code == real_code

    return {
        "excel_row": excel_row_number,
        "real_code": real_code,
        "description": description,
        "results": results,
        "vector_rank": vector_rank,
        "rerank_correct": rerank_correct,
        "best_code": best_code,
        "reason": reason,
    }


def batch_test(start: int, end: int):
    """Run rows start..end (inclusive), print accuracy summary."""
    df, _ = _load()
    max_row = len(df) + 1  # last data row (1-indexed)
    end = min(end, max_row)

    hits1 = hits5 = hits10 = rerank_hits = skipped = 0
    total = end - start + 1

    for row_num in range(start, end + 1):
        r = run_row(row_num, verbose=False)
        if not r or r["real_code"] is None:
            skipped += 1
            continue
        rank = r["vector_rank"]
        if rank == 1:
            hits1 += 1
        if 1 <= rank <= 5:
            hits5 += 1
        if 1 <= rank <= 10:
            hits10 += 1
        if r["rerank_correct"]:
            rerank_hits += 1
        rerank_marker = "✓" if r["rerank_correct"] else "✗"
        print(f"  row {row_num:4d}  real={r['real_code']}  vector={'✓'+str(rank) if rank>0 else 'miss':6s}  rerank={rerank_marker} {r['best_code']}")

    tested = total - skipped
    if tested == 0:
        print("No rows with HS codes to test.")
        return

    print(f"\n{'='*50}")
    print(f"Tested:        {tested}  (skipped {skipped} without code)")
    print(f"Vector Top-1:  {hits1}/{tested}  ({hits1/tested*100:.1f}%)")
    print(f"Vector Top-5:  {hits5}/{tested}  ({hits5/tested*100:.1f}%)")
    print(f"Vector Top-10: {hits10}/{tested}  ({hits10/tested*100:.1f}%)")
    print(f"Re-rank Top-1: {rerank_hits}/{tested}  ({rerank_hits/tested*100:.1f}%)  ← GPT pick")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage:")
        print("  python pipeline.py 112          # single row")
        print("  python pipeline.py 2 50         # batch rows 2-50")
        sys.exit(1)

    if len(args) == 1:
        run_row(int(args[0]))
    else:
        batch_test(int(args[0]), int(args[1]))
