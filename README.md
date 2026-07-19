<p align="center">
  <img src="assets/chatbi_logo_512.png" width="120" alt="ChatBI logo" />
</p>

# ChatBI

# ChatBI (NL → SQL Data Analyst Assistant)

A Streamlit app that lets a user ask business questions in plain English
("What's the total revenue by region?") and get back the generated SQL,
a result table, and a chart — powered by Groq (Llama 3.3 70B) for the
language-to-SQL step, which has a free tier with no credit card required.

## Why this project (for your resume)
- Reuses skills you already have: SQL, Pandas, dashboarding, stakeholder-facing reporting.
- Shows you can work with LLM APIs, which is a strong differentiator right now.
- Ships as a live public link, not just a GitHub repo — recruiters can actually click it.

## 1. Run it locally

```bash
cd nl2sql-chatbot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a file `.streamlit/secrets.toml` in the project folder:

```toml
GROQ_API_KEY = "gsk_your-key-here"
```

Get a free key at https://console.groq.com/keys — sign in, click "Create API Key,"
no credit card required. Check Groq's site for current rate limits on the free tier,
as these can change.

Then run:

```bash
streamlit run app.py
```

It opens at http://localhost:8501. A sample sales table is pre-loaded, or you can
upload your own CSV from the sidebar.

## 2. Deploy for free (Streamlit Community Cloud)

1. Push this folder to a public GitHub repo (keep `secrets.toml` OUT of git — add it
   to `.gitignore`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app", pick your repo, branch, and `app.py` as the entry point.
4. In "Advanced settings → Secrets", paste:
   ```toml
   GROQ_API_KEY = "gsk_your-key-here"
   ```
5. Click Deploy. You'll get a live URL — try to claim something like
   `chatbi.streamlit.app` or `chatbi-ashutosh.streamlit.app` (whatever's free) —
   put that link directly on your resume next to this project.

## 3. Ideas to extend it (good talking points in interviews)
- Swap SQLite for PostgreSQL (you already know this from D3V India / your Sales
  Dashboard project) so it's a more realistic production setup.
- Add query result caching so repeated questions don't re-call the API.
- Add a guardrail step that rejects any generated SQL containing
  `DROP`, `DELETE`, `UPDATE`, or `INSERT` (read-only analyst tool).
- Log every question + generated SQL to a table so you can show "usage analytics"
  on the chatbot itself — very on-brand for a Data Analyst project.
