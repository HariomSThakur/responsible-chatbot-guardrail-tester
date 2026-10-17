# Responsible Chatbot Guardrail Tester

A local web prototype with a Python analysis backend. It evaluates the chatbot response in the context of the original prompt and optional reference context.

## Run on Windows

Install [Ollama for Windows](https://ollama.com/download/windows) and start it. Then open PowerShell and download the free Llama 3.1 8B model (about 4.9 GB):

```powershell
ollama pull llama3.1:8b
```

In a second PowerShell window, start the app:

```powershell
cd "C:\Users\ht565\Documents\Codex\2026-10-15\cr\outputs\responsible-chatbot-guardrail-tester"
# Use regular CPython 3.13, not the free-threaded 3.13t build.
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe server.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Keep Ollama and the server window open while using the app. Stop the server with **Ctrl+C**. No API key or paid hosted inference account is required. The model runs on your computer; first use may take longer while it loads. Use regular (GIL-enabled) CPython 3.10 or newer; free-threaded builds may not have compatible dependency wheels. The project-local virtual environment keeps dependencies separate from Gradio and other Python apps.

If `.venv` was previously created with the free-threaded interpreter, rebuild that environment with `py -3.13 -m venv --clear .venv` before installing the requirements again.

## What the evaluator does

- Validates required prompt/response fields, types, and size limits. Reference context is optional.
- Uses the local Ollama service and its official Python SDK with JSON Schema structured output. Pydantic validates the returned assessment. The default is Llama 3.1 8B (`llama3.1:8b`). The model receives the original prompt, actual response, optional trusted context, and selected category focus; the focus never determines the verdict. It evaluates all supported categories and must cite exact text from the chatbot response for every finding.
- Combines the model assessment with deterministic checks for unsafe content, bias/discrimination, secrets, unsupported claims, hallucination signals, protected instructions, relevance, and toxicity.
- Distinguishes some refusals and refuted/educational quotations from harmful compliance.
- Includes the 12 supplied PDF prompt/response cases and 40 synthetic Llama practice cases in the sample selector. The practice set covers five examples for each supported category, including safe answers, clear violations, and ambiguous cases.
- Deduplicates findings by category, returns per-category risk scores, and provides a consistent JSON result with decision, weighted score, evidence, recommended actions, safer rewrite, and limitations.
- Uses `PASS`, `REVIEW`, or `FAIL`. Ambiguous cases, an unavailable local Ollama service, missing model, and invalid model output produce `REVIEW` unless an evidence-backed critical violation requires `FAIL`. Critical findings cannot be hidden by the weighted average.

Category scores are 0–100, derived from finding severity, with explicit higher scores for confirmed credential patterns and discriminatory recommendations. The overall score follows the supplied PDF's weighted dimensions: safety 30%, confidentiality 30%, bias 20%, hallucination 10%, and unsupported claims 10%. Access-control risk rolls into safety; toxicity rolls into bias; relevance rolls into unsupported claims. Only the highest score in each weighted dimension contributes, avoiding duplicate-counting overlapping categories. A critical credential leak can therefore fail while the overall score remains below 50.

## Local Llama settings

Ollama serves the model locally at `http://127.0.0.1:11434` by default. You can choose another model already installed in Ollama, and optionally change the host:

```powershell
$env:OLLAMA_MODEL = "llama3.1:8b" # optional; this is the default
$env:OLLAMA_HOST = "http://127.0.0.1:11434" # optional; this is the default
.\.venv\Scripts\python.exe server.py
```

The selected risk category is an analysis focus, not a verdict; all supported categories are still checked. Structured fields are validated, finding evidence must occur verbatim in the response, and critical findings force `FAIL`. If Ollama is stopped, the model is missing, or output cannot be validated, the dashboard shows `REVIEW` and explains why. Model output can still vary and does not guarantee safety. By default, prompts and responses are evaluated by the local Ollama service; if you configure `OLLAMA_HOST` to a remote server, that server receives the submitted content. Avoid submitting real credentials or personal information.

To try the 40 synthetic examples, choose **40 Llama practice cases** in the sample selector, select a case, then click **Run guardrail check**. The selected sample sets the risk-category focus and fills in its prompt, chatbot response, and reference context; Llama then analyzes those fields. The expected decision is shown beside the actual result as a comparison aid. The cases are stored in `tests/llm_test_cases.json` and served locally by `/api/llm/cases`.

## Adversarial tests, policies, and benchmark

The app adds four tools alongside the existing response review:

- **Adversarial tests:** The editable synthetic suite is in `tests/adversarial_cases.json`. The UI can run one selected case or the full suite. Each result shows its expected and actual decision/category, whether the test matched, and whether execution itself succeeded. Test execution errors are reported separately from safety failures.
- **Guardrail policies:** The UI edits per-category enablement, severity, risk threshold, and allow/review/block action, plus global PASS/FAIL thresholds, the critical-risk override, and uncertain-review behavior. Settings are validated on the server and saved to `policy_settings.json` in this local project folder. Reset restores defaults. Confirmed synthetic/credential patterns and protected instruction disclosure remain mandatory even if their category is disabled.
- **Evidence highlighting:** Findings return exact response spans with Unicode code-point offsets. The response view preserves the text and whitespace, supports overlapping spans, and links findings to their evidence.
- **Accuracy benchmark:** The labeled synthetic dataset is `tests/benchmark_dataset.json` (currently 24 examples: 8 safe, 8 unsafe, and 8 borderline). Metrics are computed at run time. Category metrics use one-vs-rest labels; the decision confusion matrix keeps PASS, REVIEW, and FAIL separate. For binary PASS/FAIL metrics, choose whether REVIEW is excluded, counted as positive, or counted as negative. Undefined rates are labeled when a denominator is zero.

The benchmark dataset is deliberately small and synthetic. Its results are useful for regression checks only, not as proof of production reliability. Edit the JSON datasets to add cases without changing evaluator code.

## Tests and labeled evaluation

From the project folder:

```powershell
python -m unittest discover -s tests -v
python tests/evaluate_dataset.py
# Optional: choose how binary decision metrics treat REVIEW
python tests/evaluate_dataset.py --review-handling positive
```

The test suite covers existing PDF cases, Ollama structured output configuration, category-focus transmission, safe and harmful answers to the same dangerous prompt, refusals, ambiguous/error cases, evidence validation, rule detections, policy validation, adversarial execution, benchmark metrics, and API errors. Ollama responses are mocked only in unit tests; production uses the configured local Ollama service.

## Privacy and limitations

The server only binds to a loopback address (`127.0.0.1`, `localhost`, or `::1`), rejects cross-origin browser requests, accepts JSON POST bodies up to 60 KB, and does not log request URLs or submitted text. Responses include a restrictive Content Security Policy and browser security headers. This is still a local hackathon prototype, not an authenticated service for a shared or internet-facing deployment. Avoid entering real credentials or personal information. Factual accuracy and hallucinations are not verified without suitable reference context or an independent source. Confidence values are estimates of evidence strength, not calibrated probabilities. Human review is still needed for release decisions.
