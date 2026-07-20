import streamlit as st
import pandas as pd
import sqlite3
import re
import os
from groq import Groq

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="ChatBI", page_icon="assets/favicon.ico", layout="wide")

import base64
with open("assets/chatbi_logo_200.png", "rb") as f:
    logo_b64 = base64.b64encode(f.read()).decode()

st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:0.5rem;">
        <img src="data:image/png;base64,{logo_b64}" width="55">
        <h1 style="margin:0; padding:0;">ChatBI</h1>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Ask questions in plain English. ChatBI converts them to SQL, runs them, and shows results + a chart.")

# ---------- DB SETUP ----------
DB_PATH = "sales.db"

@st.cache_resource
def init_db():
    """Create a sample sales table the first time the app runs."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY,
            region TEXT,
            product TEXT,
            category TEXT,
            revenue REAL,
            profit REAL,
            order_date TEXT
        )
    """)
    cur.execute("SELECT COUNT(*) FROM sales")
    if cur.fetchone()[0] == 0:
        sample = pd.DataFrame({
            "region": ["North", "South", "East", "West"] * 25,
            "product": ["Widget A", "Widget B", "Widget C", "Widget D", "Widget E"] * 20,
            "category": ["Electronics", "Home", "Electronics", "Toys", "Home"] * 20,
            "revenue": [round(x, 2) for x in (pd.Series(range(100)) * 37.5 + 200)],
            "profit": [round(x, 2) for x in (pd.Series(range(100)) * 9.1 + 30)],
            "order_date": pd.date_range("2025-01-01", periods=100, freq="3D").strftime("%Y-%m-%d"),
        })
        sample.to_sql("sales", conn, if_exists="append", index=False)
    conn.commit()
    return conn

conn = init_db()

# ---------- SIDEBAR: LET USER UPLOAD THEIR OWN CSV TOO ----------
st.sidebar.header("Data source")
uploaded = st.sidebar.file_uploader("Upload your own CSV (optional)", type=["csv"])
if uploaded:
    df_uploaded = pd.read_csv(uploaded)
    df_uploaded.to_sql("sales", conn, if_exists="replace", index=False)
    st.sidebar.success(f"Loaded {len(df_uploaded)} rows into the 'sales' table.")

schema_df = pd.read_sql("PRAGMA table_info(sales)", conn)
st.sidebar.subheader("Current table schema")
st.sidebar.dataframe(schema_df[["name", "type"]], hide_index=True)

# ---------- GROQ CLIENT (free tier) ----------
# Put your key in .streamlit/secrets.toml as GROQ_API_KEY = "gsk_..."
# or set it as an environment variable before running.
# Get a free key at https://console.groq.com/keys — no credit card required.
api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
client = Groq(api_key=api_key) if api_key else None

def nl_to_sql(question: str, schema: pd.DataFrame, previous_sql: str = None, previous_error: str = None) -> str:
    """Ask a Groq-hosted model to turn a natural-language question into a SQLite query.
    If previous_sql/previous_error are given, the model is asked to fix its earlier mistake."""
    columns = ", ".join(f"{r['name']} ({r['type']})" for _, r in schema.iterrows())
    if previous_sql and previous_error:
        prompt = f"""You are a SQL expert. Table name: sales. Columns: {columns}.
The question is: "{question}"
Your previous SQL attempt failed:
SQL: {previous_sql}
Error: {previous_error}
Write ONE corrected SQLite query that fixes this error and answers the question.
Rules: return ONLY the SQL query, no explanation, no markdown fences."""
    else:
        prompt = f"""You are a SQL expert. Table name: sales. Columns: {columns}.
Write ONE SQLite query that answers this question: "{question}"
Rules: return ONLY the SQL query, no explanation, no markdown fences, no semicolon-terminated comments."""
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    sql = resp.choices[0].message.content.strip()
    sql = re.sub(r"^```sql|```$", "", sql, flags=re.MULTILINE).strip()
    return sql

# ---------- GUARDRAIL ----------
FORBIDDEN_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "REPLACE", "ATTACH", "PRAGMA"]

def is_safe_sql(sql: str) -> bool:
    """Reject any query containing destructive or non-read-only keywords."""
    upper_sql = sql.upper()
    return not any(re.search(rf"\b{kw}\b", upper_sql) for kw in FORBIDDEN_KEYWORDS)

# ---------- RETRY-ON-ERROR EXECUTION ----------
def run_query_with_retry(question: str, schema: pd.DataFrame, conn, max_retries: int = 1):
    """Generate SQL, validate it, run it, and retry once with the error fed back to the model if it fails."""
    sql = nl_to_sql(question, schema)
    last_error = None
    for attempt in range(max_retries + 1):
        if not is_safe_sql(sql):
            return sql, None, "Blocked: generated SQL contained a non-read-only statement."
        try:
            df = pd.read_sql(sql, conn)
            return sql, df, None
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                sql = nl_to_sql(question, schema, previous_sql=sql, previous_error=last_error)
    return sql, None, last_error

# ---------- RESULT SUMMARY (uses a smaller/faster model — cheaper for a simple task) ----------
def summarize_result(question: str, df: pd.DataFrame) -> str:
    preview = df.head(20).to_csv(index=False)
    prompt = f"""Question asked: "{question}"
Result data (CSV, up to 20 rows):
{preview}
Write a 2-3 sentence plain-English summary of what this data shows. Mention the key number(s) or trend. No preamble, just the summary."""
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()

# ---------- CHAT UI ----------
if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("e.g. What is the total revenue by region?")

if question:
    st.session_state.history.append(("user", question))
    if not client:
        sql, result, error, summary = None, None, "No GROQ_API_KEY found. Add it to .streamlit/secrets.toml to enable the chatbot.", None
    else:
        with st.spinner("Thinking..."):
            sql, result, error = run_query_with_retry(question, schema_df, conn)
            summary = None
            if result is not None and not result.empty:
                try:
                    summary = summarize_result(question, result)
                except Exception:
                    summary = None
    st.session_state.history.append(("assistant", sql, result, error, summary))

for entry in st.session_state.history:
    if entry[0] == "user":
        with st.chat_message("user"):
            st.write(entry[1])
    else:
        _, sql, result, error, summary = entry
        with st.chat_message("assistant"):
            if error:
                st.error(error)
            if sql:
                st.code(sql, language="sql")
            if result is not None:
                if summary:
                    st.markdown(f"**Summary:** {summary}")
                st.dataframe(result, use_container_width=True)
                numeric_cols = result.select_dtypes("number").columns
                if len(result.columns) >= 2 and len(numeric_cols) >= 1:
                    label_col = [c for c in result.columns if c not in numeric_cols][0] if len(result.columns) > len(numeric_cols) else result.columns[0]
                    st.bar_chart(result, x=label_col, y=numeric_cols[0])

st.divider()
st.caption("ChatBI — built with Streamlit + Groq API (Llama 3.3) + SQLite. Portfolio project by Ashutosh Shinde.")