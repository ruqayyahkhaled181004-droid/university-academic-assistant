# Campus Guide — University Academic Assistant

Start with University_Academic_Assistant_Gemini.ipynb in Google Colab. Run numbered cells from top to bottom.
Use a CPU runtime. Add GEMINI_API_KEY to Colab Secrets and enable notebook access, or use the masked prompt.
Get a key at https://aistudio.google.com/apikey. Free-tier access depends on your account, region and quota.
Paid billing is optional where free-tier access is available; a consumer Gemini subscription does not determine API quota.

## Project scope
Students can ask questions about academic regulations, courses, registration, attendance, exams and graduation.
The application retrieves document passages, generates cited answers, resolves follow-ups, classifies questions,
and records prompt versions, latency, tokens, estimated cost and error types.
It does not access student systems, change grades, make official eligibility decisions, or invent missing policies.

## Data
Included DEMO_*.txt files are fictional educational examples, not AASTMT or other real university rules.
To use real data choose USE_DEMO=False in the notebook, upload official PDFs, and revise the evaluation dataset.
Use documents for one institution/program/year or include that scope in your questions. Scanned PDFs need OCR.
Uploaded document text and questions are sent to Gemini. Operational logs omit question and document text.
Evaluation CSVs deliberately include test questions/answers; use public or synthetic evaluation content.

## Run Streamlit
pip install -r requirements.txt
streamlit run app.py

## Deploy for final submission
1. Create a GitHub repository and upload app.py, rag.py, requirements.txt, README.md and .gitignore from this folder.
2. Open https://share.streamlit.io and choose Create app; select the repository, branch and app.py.
3. In the app's Secrets settings add GEMINI_API_KEY = "your-key". Never put a real key in a repository or notebook cell.
4. Deploy, upload your documents and build the index. Test a normal question, a follow-up and an unsupported question.
5. Save the live URL for submission. This package does not create a hosted deployment automatically.
The app also accepts a visitor's own key if no server secret is configured. A public app using your server key
can spend your API budget; use deployment access controls and account spend limits for your demonstration.
Indexes and conversations are per session. Hosted storage is ephemeral; download logs before restart.

## Evaluation and limitations
The notebook compares v1 and v2 on identical questions and retrieval settings. Do not assume v2 wins.
The optional Cell 12B uses the course LLM-as-judge technique to suggest groundedness and citation-support
scores. It is disabled by default to save quota. It uses the same model and can share its biases; a human
must still review claims. The Streamlit monitoring dashboard uses the same session event logs.
Automated answer checks are keyword proxies; valid citation labels do not prove a claim is supported.
Complete the manual groundedness and citation-support columns by checking each claim against its passage.
The guard combines a small regex filter, instruction separation, prompt rules and citation validation.
It is a classroom baseline, not a guarantee against every prompt injection or hallucination.
Latency includes follow-up rewriting, query embedding, answer generation, quota pacing and retries. Calls
are spaced 13 seconds apart per model/client by default; this is not a guarantee of fitting every account quota.
Change min_interval_s when constructing AcademicAssistant if your actual account limit requires it.
Cost is an estimate using
published standard paid-tier rates checked 2026-09-10, ignoring cached discounts. Free-tier billed cost may be zero.
When embedding responses omit token usage, characters/4 is a rough estimate (especially approximate for non-English text),
flagged by embedding_tokens_estimated. Missing generation usage is flagged separately; cost can then be incomplete. API retry billing is not fully observable.
Document-index embedding cost is logged separately. No real API results are pre-filled.

## Final demonstration
Explain problem/users → build index → inspect retrieved passages → ask a cited question → ask a follow-up
→ show unsupported and adversarial handling → compare evaluation → inspect logs → open deployed app.
Before submission: replace demo data if needed, run evaluation, manually review citations, record observed
before/after results, and deploy. A working notebook alone does not satisfy the deployment requirement.

## Official references
- https://ai.google.dev/gemini-api/docs/quickstart
- https://ai.google.dev/gemini-api/docs/pricing
- https://ai.google.dev/gemini-api/docs/embeddings
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
