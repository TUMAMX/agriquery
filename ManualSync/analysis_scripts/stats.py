import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

B = 2000
SEED = 20260926
DATASETS = {"general_questions": "answerable", "unanswerable_questions": "unanswerable"}
REFUSAL = re.compile(
    r"not found|unknown|\bn/a\b|cannot be determined|"
    r"\bno (?:specific |explicit |such |relevant |direct |clear )?(?:information|details?|data|mention|materials?|precautions?|"
    r"system|instructions?|answer|reference|value|procedure|guidance)\b[^.]{0,80}?(?:available|provided|mentioned|identified|given|"
    r"found|specified|stated|included|present|contained|listed)|"
    r"\b(?:is|are|was|were) not (?:explicitly )?(?:mentioned|specified|provided|available|stated|included|given|present|identified|listed|found)|"
    r"\bno (?:specific |explicit |such |particular )\w+(?: \w+)? (?:is|are|was|were) (?:explicitly )?(?:mentioned|specified|provided|given|found|identified|listed|stated|indicated)|"
    r"does not (?:explicitly )?(?:mention|specify|provide|contain|say|state|include)|isn't (?:mentioned|specified|provided)|"
    r"\bnon (?:è|e|viene) (?:specificat|indicat|menzionat|riportat)|\bno se (?:ha |)?(?:especifica|menciona|indica|encuentra|detall)|"
    r"nicht (?:angegeben|erwähnt|spezifiziert|enthalten)|niet (?:vermeld|gespecificeerd|gevonden|aangegeven)|"
    r"n'est pas (?:mentionn|précis|indiqu|spécifi)",
    re.I)
REFUSAL_WINDOW = 160
PLACEHOLDER = re.compile(r"\[Carefully extract the EXACT information.*?\]", re.S)


def final_answer(text):
    t = text or ""
    if "</think>" in t:
        t = t.split("</think>")[-1]
    return PLACEHOLDER.sub("", t).strip()


def is_refusal(text):
    fa = re.sub(r"</?answer>|[*_]", "", final_answer(text))
    fa = re.sub(r"^\W+", "", fa.strip())
    return bool(REFUSAL.search(fa[:REFUSAL_WINDOW]))


FOREIGN = {"de", "fr", "nl", "it", "es"}


def load_answer_languages(csv_paths):
    out = {}
    for path in csv_paths or []:
        for r in csv.DictReader(open(path, encoding="utf-8")):
            out[(r["file"], r["dataset"], int(r["index"]))] = r["answer_language"]
    return out


def load_results(dirs, answer_lang=None):
    conds = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.json")):
            x = json.loads(f.read_text(encoding="utf-8"))
            if "test_run_parameters" not in x:
                continue
            p = x["test_run_parameters"]
            q, cor, ans, cat = [], [], [], []
            excluded = 0
            for ds, kind in DATASETS.items():
                for idx, r in enumerate(x["per_dataset_details"].get(ds, {}).get("results", [])):
                    if r.get("qa_error") or r.get("eval_error") or r.get("self_evaluation") not in ("yes", "no"):
                        excluded += 1
                        continue
                    ok = r.get("self_evaluation") == "yes"
                    refused = is_refusal(r.get("model_answer"))
                    foreign = answer_lang is not None and answer_lang.get((f.name, ds, idx)) in FOREIGN
                    if answer_lang is not None and (f.name, ds, idx) not in answer_lang:
                        raise KeyError(f"no answer language for {f.name} {ds} {idx}")
                    q.append(r["question"])
                    cor.append(ok and not foreign)
                    ans.append(kind == "answerable")
                    if foreign:
                        cat.append("foreign_language")
                    elif kind == "answerable":
                        cat.append("correct" if ok else ("false_refusal" if refused else "wrong_answer"))
                    else:
                        cat.append("correct_abstention" if ok else "hallucination")
            conds.append({
                "model": p["question_model"], "language": p.get("language", p.get("language_tested", "")).replace("_manual", ""),
                "format": p.get("file_extension", "md"), "retrieval": p["retrieval_algorithm"], "judge": p["evaluator_model"],
                "source": f"{Path(d).name}/{f.name}", "questions": q, "correct": np.array(cor, bool),
                "answerable": np.array(ans, bool), "category": cat, "reported_f1": x["overall_metrics"]["f1_score"],
                "excluded_no_verdict": excluded,
            })
    return conds


def metrics_from_counts(tp, fn, tn, fp):
    tp, fn, tn, fp = (np.asarray(v, float) for v in (tp, fn, tn, fp))
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        spec = np.where(tn + fp > 0, tn / (tn + fp), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        acc = (tp + tn) / (tp + fn + tn + fp)
    return {"accuracy": acc, "precision": prec, "recall": rec, "specificity": spec, "f1": f1}


def counts(correct, answerable, weights=None):
    w = np.ones(len(correct)) if weights is None else weights
    tp = (w * (correct & answerable)).sum(-1)
    fn = (w * (~correct & answerable)).sum(-1)
    tn = (w * (correct & ~answerable)).sum(-1)
    fp = (w * (~correct & ~answerable)).sum(-1)
    return tp, fn, tn, fp


def bootstrap_weights(question_ids, answerable_by_q, rng, b=B):
    ids = np.array(sorted(set(question_ids)))
    ans = np.array([answerable_by_q[i] for i in ids])
    w = np.zeros((b, len(ids)))
    for stratum in (True, False):
        idx = np.where(ans == stratum)[0]
        draws = rng.integers(0, len(idx), size=(b, len(idx)))
        for k in range(b):
            np.add.at(w[k], idx[draws[k]], 1)
    return ids, w


def pooled_arrays(conds):
    qid, cor, ans = [], [], []
    for c in conds:
        qid += c["questions"]
        cor.append(c["correct"])
        ans.append(c["answerable"])
    return np.array(qid), np.concatenate(cor), np.concatenate(ans)


def summarize(conds, rng):
    qid, cor, ans = pooled_arrays(conds)
    point = metrics_from_counts(*counts(cor, ans))
    ids, w = bootstrap_weights(qid, dict(zip(qid, ans)), rng)
    pos = {q: i for i, q in enumerate(ids)}
    obs_w = w[:, [pos[q] for q in qid]]
    boot = metrics_from_counts(*counts(cor[None, :], ans[None, :], obs_w))
    out = {"n_obs": len(cor), "n_questions": len(ids)}
    for k in point:
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        out[k] = float(point[k]); out[k + "_lo"] = float(lo); out[k + "_hi"] = float(hi)
    cats = {}
    for c in conds:
        for v in c["category"]:
            cats[v] = cats.get(v, 0) + 1
    out.update({f"n_{k}": cats.get(k, 0) for k in
                ["correct", "wrong_answer", "false_refusal", "foreign_language", "correct_abstention", "hallucination"]})
    return out


def _keyed(conds, key_fields):
    obs = {}
    for c in conds:
        for q, ok, an in zip(c["questions"], c["correct"], c["answerable"]):
            obs[tuple(c[f] for f in key_fields if f != "question") + (q,)] = (bool(ok), bool(an))
    return obs


def paired(conds_a, conds_b, rng, key_fields=("language", "question")):
    ka, kb = _keyed(conds_a, key_fields), _keyed(conds_b, key_fields)
    keys = sorted(set(ka) & set(kb))
    ca = np.array([ka[k][0] for k in keys]); cb = np.array([kb[k][0] for k in keys])
    aa = np.array([ka[k][1] for k in keys]); qid = np.array([k[-1] for k in keys])
    f1a = float(metrics_from_counts(*counts(ca, aa))["f1"]); f1b = float(metrics_from_counts(*counts(cb, aa))["f1"])
    ids, w = bootstrap_weights(qid, dict(zip(qid, aa)), rng)
    pos = {q: i for i, q in enumerate(ids)}
    obs_w = w[:, [pos[q] for q in qid]]
    d = (metrics_from_counts(*counts(ca[None, :], aa[None, :], obs_w))["f1"]
         - metrics_from_counts(*counts(cb[None, :], aa[None, :], obs_w))["f1"])
    lo, hi = np.percentile(d, [2.5, 97.5])
    p_boot = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    b_only = int((ca & ~cb).sum()); c_only = int((cb & ~ca).sum())
    p_mcn = binomtest(b_only, b_only + c_only, 0.5).pvalue if b_only + c_only else 1.0
    return {"n_pairs": len(keys), "f1_a": f1a, "f1_b": f1b, "delta_f1": f1a - f1b, "delta_lo": float(lo), "delta_hi": float(hi),
            "p_bootstrap": p_boot, "a_only_correct": b_only, "b_only_correct": c_only, "p_mcnemar": float(p_mcn)}


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def per_condition_table(conds, rng):
    rows = []
    for c in conds:
        s = summarize([c], rng)
        rows.append({"model": c["model"], "language": c["language"], "format": c["format"], "retrieval": c["retrieval"],
                     "judge": c["judge"], **s, "reported_f1": c["reported_f1"], "source": c["source"]})
    return rows


REFUSAL_TESTS = [
    ("Not found in context", True), ("Unknown", True), ("50 Nm", False),
    ("[No specific information provided in the context to answer the question.]", True),
    ("<answer>     [No such system is identified in the provided context.] </answer>", True),
    ("The required distance is **not explicitly provided** in the context.", True),
    ("**Answer:**   *No specific symbol is mentioned in the context to indicate the kit is enabled.*", True),
    ("The decoupling ropes must not be taut.", False),
    ("Do not stand between the tractor and the machine during coupling.", False),
]

LANGS = ["english", "german", "french", "dutch", "italian", "spanish"]


def pooled_table(conds, rng, group_fields=("model", "retrieval", "format")):
    groups = {}
    for c in conds:
        groups.setdefault(tuple(c[f] for f in group_fields), []).append(c)
    rows = []
    for key, cs in sorted(groups.items()):
        for label, sel in [("EN", [c for c in cs if c["language"] == "english"]),
                           ("non-EN", [c for c in cs if c["language"] != "english"])]:
            if sel:
                rows.append({**dict(zip(group_fields, key)), "languages": label, "n_conditions": len(sel), **summarize(sel, rng)})
    return rows


def comparison_family(conds, rng, pairs, group_field="model", family="primary"):
    rows = []
    for g in sorted({c[group_field] for c in conds}):
        cg = [c for c in conds if c[group_field] == g]
        for field, va, vb in pairs:
            for label, langsel in [("EN", ["english"]), ("non-EN", LANGS[1:])]:
                a = [c for c in cg if c[field] == va and c["language"] in langsel]
                b = [c for c in cg if c[field] == vb and c["language"] in langsel]
                if a and b:
                    r = paired(a, b, rng)
                    rows.append({group_field: g, "contrast": f"{va} vs {vb}", "languages": label, "family": family, **r})
    for key in ("p_mcnemar", "p_bootstrap"):
        adj = holm([r[key] for r in rows])
        for r, v in zip(rows, adj):
            r[key + "_holm"] = float(v)
    return rows


def selftest(rng):
    bad = [(t, e) for t, e in REFUSAL_TESTS if is_refusal(t) != e]
    assert not bad, f"refusal pattern fails on: {bad}"
    print(f"refusal unit tests: {len(REFUSAL_TESTS)} passed")
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ablation", action="store_true", help="also compute the primary ablation comparison family")
    ap.add_argument("--formats", action="store_true", help="also compute the format comparison family (md/json/xml)")
    ap.add_argument("--strict-lang", nargs="*", default=None,
                    help="answer-language CSV(s); answers in the manual's language count as incorrect (strict rule)")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    if a.selftest:
        selftest(rng)
        return
    conds = load_results(a.results, load_answer_languages(a.strict_lang) if a.strict_lang else None)
    out = Path(a.out)
    write_csv(out / "per_condition.csv", per_condition_table(conds, rng))
    write_csv(out / "pooled_EN_vs_nonEN.csv", pooled_table(conds, rng))
    print(f"{len(conds)} conditions -> {out}/per_condition.csv, pooled_EN_vs_nonEN.csv")
    if a.ablation:
        pairs = [("retrieval", "hybrid", "keyword"), ("retrieval", "hybrid", "embedding"),
                 ("retrieval", "embedding", "keyword")]
        pairs += [("retrieval", "hybrid", v) for v in ("no_context",) if any(c["retrieval"] == v for c in conds)]
        pairs += [("retrieval", "oracle", "hybrid")] if any(c["retrieval"] == "oracle" for c in conds) else []
        write_csv(out / "comparisons_primary.csv", comparison_family(conds, rng, pairs))
        print(f"primary comparisons ({len(pairs)} contrasts x models x EN/non-EN) -> {out}/comparisons_primary.csv")
    if a.formats:
        pairs = [("format", "md", "json"), ("format", "md", "xml"), ("format", "json", "xml")]
        write_csv(out / "comparisons_formats.csv", comparison_family(conds, rng, pairs, family="formats"))
        print(f"format comparisons ({len(pairs)} contrasts x models x EN/non-EN) -> {out}/comparisons_formats.csv")
        write_csv(out / "pooled_by_format.csv", pooled_table(conds, rng, group_fields=("format",)))
        print(f"all models pooled per format (EN / non-EN) -> {out}/pooled_by_format.csv")


if __name__ == "__main__":
    main()
