# ManualSync

ManualSync is a retrieval-augmented generation (RAG) pipeline that answers English questions from operator manuals of
agricultural machinery with locally run open-weight LLMs. It processes manuals in several languages and formats (Markdown,
JSON, XML) and scores the answers with an LLM judge.

## Contents

| Path | Content |
|---|---|
| `main.py` | entry point: builds the vector databases and runs question answering and judging as set in `config.json` |
| `config.json` | models, manuals, formats, retrieval parameters and prompts |
| `rag_pipeline.py`, `rag_tester.py` | pipeline and test runner |
| `retrieval_pipelines/` | BM25 (`keyword`), semantic (`embedding`) and hybrid retrieval (Reciprocal Rank Fusion) |
| `llm_connectors/` | connection to Ollama |
| `evaluation/` | LLM judge and metrics |
| `prompt_templates/` | question and judge prompts |
| `question_datasets/` | 108 questions (54 answerable, 54 unanswerable) with expected answers and the page of the source passage |
| `judge_validation/rating_sheet.md` | items for a human validation of the LLM judge |
| `analysis_scripts/` | answer language, statistics and retrieval metrics (see below) |
| `plot_scripts/`, `visualization/` | plotting code |
| `scripts/` | helper scripts, e.g. a quick test of the Ollama models (see `scripts/README.md`) |

## Manuals (not included)

The operator manuals are copyrighted by the manufacturer and are not included. The questions refer to the public PDF editions
of the operator manual of the Kverneland Exacta-TLX GEOSPREAD GS3:
[English](https://www.kvgportal.com/W_global/Media/lexcom/VN/A14870/A148703540-2.pdf),
[German](https://www.kvgportal.com/W_global/Media/lexcom/VN/A14880/A148818240-1.pdf),
[French](https://www.kvgportal.com/W_global/Media/lexcom/VN/A14870/A148703640-2.pdf),
[Dutch](https://www.kvgportal.com/W_global/Media/lexcom/VN/A14870/A148703740-2.pdf),
[Italian](https://www.kvgportal.com/W_global/Media/lexcom/VN/A14880/A148818540.pdf),
[Spanish](https://www.kvgportal.com/W_global/Media/lexcom/VN/A14880/A148818340.pdf).

Convert each PDF page by page with the Docling-based converter in
[`../agri-query/ZeroShot/docling_page_wise_pdf_converter/`](../agri-query/ZeroShot/docling_page_wise_pdf_converter/) and place
the files in `manuals/` as `<language>_manual.<format>`, e.g. `english_manual.md`, `german_manual.json`, `dutch_manual.xml`.

## Installation

Requirements: Python 3.12, [Ollama](https://ollama.com), and enough memory for the embedding model (Qwen3-Embedding-8B).

```bash
git clone https://github.com/TUMAMX/agriquery.git
cd agriquery/ManualSync
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Pull the Ollama models listed in `config.json` (`llm_models.ollama`, e.g. `ollama pull qwen2.5:latest`).

## Configuration

Default settings in `config.json`:

| Setting | Value |
|---|---|
| Question models (`question_models_to_test`) | GPT-OSS 120B, Gemma 3 12B, Qwen 3 8B, DeepSeek-R1 8B, Qwen 2.5 7B, Llama 3.1 8B, Llama 3.2 3B, DeepSeek-R1 1.5B |
| Judge (`evaluator_model_name`) | GPT-OSS 20B |
| Manuals and formats | six language editions; `md`, `json`, `xml` |
| Embedding model | Qwen/Qwen3-Embedding-8B |
| Retrieval | `hybrid`; chunks of 200 tokens with 100 tokens overlap; top 3 chunks |

`rag_parameters.retrieval_algorithms_to_test` accepts `keyword` (BM25), `embedding` (semantic), `hybrid`, `no_context`
(no context) and `oracle` (the page of the source passage as context).

## Usage

```bash
python main.py
```

`main.py` builds the vector databases (ChromaDB) and runs question answering and judging for all combinations in
`config.json`. The results are written as JSON files to `results/`, with the run parameters, the overall metrics and the
answer and verdict for every question.

## Analysis

The scripts in `analysis_scripts/` work on the result files in `results/`:

```bash
python analysis_scripts/answer_language.py results answer_language.csv
python analysis_scripts/stats.py --results results --out stats --strict-lang answer_language.csv --formats
python analysis_scripts/retrieval_metrics.py chroma_db retrieval_metrics
```

- `answer_language.py` detects the language of every answer.
- `stats.py` computes accuracy, precision, recall, specificity and F1 with question-level bootstrap confidence intervals and
  paired comparisons with Holm correction (`--formats`: Markdown, JSON and XML; `--ablation`: retrieval conditions). With
  `--strict-lang`, answers in the language of the manual instead of English count as incorrect.
- `retrieval_metrics.py` computes Recall@3 and MRR@10 of BM25, semantic and hybrid retrieval for the answerable questions
  (no LLM needed).

## License

See [LICENSE](../LICENSE).
