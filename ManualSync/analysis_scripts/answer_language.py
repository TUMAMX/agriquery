import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from lingua import Language, LanguageDetectorBuilder

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
OUT_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("answer_language.csv")
MIN_CONF = 0.50
MIN_MARGIN = 0.15
LANGS = {Language.ENGLISH: "en", Language.GERMAN: "de", Language.FRENCH: "fr", Language.DUTCH: "nl",
         Language.ITALIAN: "it", Language.SPANISH: "es"}
DET = LanguageDetectorBuilder.from_languages(*LANGS).build()


def clean(text):
    t = re.sub(r"[-]", " ", str(text))
    t = re.sub(r"</?answer>", " ", t)
    t = re.sub(r"\[Carefully extract[^\]]*\]", " ", t)
    return " ".join(t.split())


def classify(text):
    t = clean(text)
    words = re.findall(r"[^\W\d_]{2,}", t)
    if len(words) < 3:
        if re.search(r"[äöüßàâçéèêëîïôùûñ¿¡]", t, re.I):
            return "mixed", None, t
        if len(words) == 2:
            c = DET.compute_language_confidence_values(t)
            if c[0].language != Language.ENGLISH and c[0].value >= 0.9 and c[0].value - c[1].value >= 0.5:
                return "mixed", round(c[0].value, 3), t
        return "neutral", None, t
    conf = DET.compute_language_confidence_values(t)
    best, second = conf[0], conf[1]
    if best.language == Language.ENGLISH:
        for sent in re.split(r"[.!?;:\n()\[\]\"“”]+", t):
            if len(re.findall(r"[^\W\d_]{2,}", sent)) < 3:
                continue
            c = DET.compute_language_confidence_values(sent)
            if c[0].language != Language.ENGLISH and c[0].value >= MIN_CONF and c[0].value - c[1].value >= MIN_MARGIN:
                return "mixed", round(best.value, 3), t
        return "en", round(best.value, 3), t
    if best.value >= MIN_CONF and best.value - second.value >= MIN_MARGIN:
        return LANGS[best.language], round(best.value, 3), t
    return "mixed", round(best.value, 3), t


def main():
    rows = []
    for f in sorted(SRC.glob("*_manual_*.json")):
        x = json.loads(f.read_text(encoding="utf-8"))
        p = x["test_run_parameters"]
        for ds, v in x["per_dataset_details"].items():
            for i, r in enumerate(v["results"]):
                lab, conf, t = classify(r.get("model_answer", ""))
                rows.append({"file": f.name, "dataset": ds, "index": i, "model": p["question_model"],
                             "edition": p["language"].replace("_manual", ""), "format": p.get("file_extension"),
                             "judge": r.get("self_evaluation"), "answer_language": lab, "confidence": conf,
                             "answer_clean": t[:300]})
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    c = Counter(r["answer_language"] for r in rows)
    print(f"{n} answers:", {k: f"{v} ({v / n:.1%})" for k, v in c.most_common()})
    foreign = [r for r in rows if r["answer_language"] not in ("en", "neutral", "mixed")]
    same = sum(r["answer_language"] == {"german": "de", "french": "fr", "dutch": "nl", "italian": "it", "spanish": "es",
                                        "english": "en"}[r["edition"]] for r in foreign)
    print(f"non-English answers: {len(foreign)}; in the language of the manual edition: {same}")
    print("  by model:", Counter(r["model"] for r in foreign).most_common())
    print("  by edition:", Counter(r["edition"] for r in foreign).most_common())
    print("  by dataset x judge verdict:", Counter((r["dataset"][:6], r["judge"]) for r in foreign).most_common())
    print(f"-> {OUT_FILE}")


if __name__ == "__main__":
    main()
