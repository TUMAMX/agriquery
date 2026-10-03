import bisect
import csv
import json
import re
import sys
from pathlib import Path

import chromadb
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from retrieval_pipelines.embedding_retriever import EmbeddingRetriever
from retrieval_pipelines.keyword_retrieval import KeywordRetriever

LANGS = ["english", "german", "french", "dutch", "italian", "spanish"]
FORMATS = ["md", "json", "xml"]
DB_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "chroma_db"
MARKER = {"md": r"(?m)^## Page (\d+)\s*$", "xml": r'<page number="(\d+)">', "json": r'"page_number": (\d+)'}
FETCH_K, TOP_K, RRF_K, B, SEED = 10, 3, 60, 2000, 20260926
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("retrieval_metrics")


def chunk_pages(text, chunks, fmt):
    marks = [(m.start(), int(m.group(1))) for m in re.finditer(MARKER[fmt], text)]
    pos = [p for p, _ in marks]
    out, start_from = [], 0
    for c in chunks:
        i = text.find(c)
        if i < 0:
            out.append(None)
            continue
        j = text.find(c, i + 1)
        spans = [(i, i + len(c))] + ([(j, j + len(c))] if j >= 0 else [])
        pages = set()
        for a, b in spans:
            k = bisect.bisect_right(pos, a) - 1
            if k >= 0:
                pages.add(marks[k][1])
            for p, n in marks:
                if a <= p < b:
                    pages.add(n)
        out.append(pages)
    return out


def rrf(sem, kw):
    ranks = {}
    for lst in (sem, kw):
        for r, d in enumerate(lst):
            ranks[d] = ranks.get(d, 0.0) + 1.0 / (RRF_K + r + 1)
    return sorted(ranks, key=lambda d: ranks[d], reverse=True)


def metrics(ranked_pages, gold):
    hit3 = any(gold in p for p in ranked_pages[:TOP_K])
    rr = next((1.0 / (r + 1) for r, p in enumerate(ranked_pages[:FETCH_K]) if gold in p), 0.0)
    return hit3, rr


def ci(values, rng):
    v = np.asarray(values, float)
    boot = v[rng.integers(0, len(v), size=(B, len(v)))].mean(1)
    return float(v.mean()), *map(float, np.percentile(boot, [2.5, 97.5]))


def main():
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    qs = json.loads((HERE / cfg["question_dataset_paths"]["general_questions"]).read_text(encoding="utf-8"))
    print(f"{len(qs)} answerable questions; loading embedding model (CPU) ...", flush=True)
    emb = EmbeddingRetriever(model_config=cfg["embedding_model"])
    qvec = [emb.vectorize_query(q["question"]) for q in qs]
    print("query embeddings done", flush=True)
    rng = np.random.default_rng(SEED)
    rows, perq = [], []
    for fmt in FORMATS:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        for lang in LANGS:
            col = client.get_collection(f"{lang}_manual_{fmt}_cs200_os100")
            docs = col.get(include=["documents"])["documents"]
            text = open(HERE / "manuals" / f"{lang}_manual.{fmt}", encoding="utf-8").read()
            pmap = dict(zip(docs, chunk_pages(text, docs, fmt)))
            unmapped = sum(v is None for v in pmap.values())
            kw = KeywordRetriever()
            kw.build_index(docs)
            res = {m: {"hit3": [], "rr": []} for m in ("bm25", "semantic", "hybrid")}
            for q, v in zip(qs, qvec):
                sem = col.query(query_embeddings=v, n_results=FETCH_K, include=["documents"])["documents"][0]
                bm = kw.retrieve_relevant_chunks(query_representation=kw.vectorize_query(q["question"]), top_k=FETCH_K)[0]
                hyb = rrf(sem, bm)
                for m, lst in (("bm25", bm), ("semantic", sem), ("hybrid", hyb)):
                    h, rr = metrics([pmap.get(d) or set() for d in lst], int(q["page"]))
                    res[m]["hit3"].append(h)
                    res[m]["rr"].append(rr)
                    perq.append({"format": fmt, "language": lang, "method": m, "question": q["question"][:80],
                                 "gold_page": q["page"], "hit_at_3": int(h), "reciprocal_rank": round(rr, 4)})
            for m, r in res.items():
                rec, rlo, rhi = ci(r["hit3"], rng)
                mrr, mlo, mhi = ci(r["rr"], rng)
                rows.append({"format": fmt, "language": lang, "method": m, "recall_at_3": rec, "recall_lo": rlo,
                             "recall_hi": rhi, "mrr_at_10": mrr, "mrr_lo": mlo, "mrr_hi": mhi, "n_questions": len(qs),
                             "n_chunks": len(docs), "unmapped_chunks": unmapped})
            print(f"  {fmt:4} {lang:8} chunks={len(docs):5} unmapped={unmapped:3}  "
                  + "  ".join(f"{m} R@3={np.mean(r['hit3']):.2f} MRR={np.mean(r['rr']):.2f}" for m, r in res.items()), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("retrieval_metrics.csv", rows), ("per_question.csv", perq)):
        with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()} for d in data])
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
