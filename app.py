import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import re
import os
from groq import Groq
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

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

def nl_to_sql(question: str, schema: pd.DataFrame, previous_sql: str = None, previous_error: str = None, context: list = None) -> str:
    """Ask a Groq-hosted model to turn a natural-language question into a SQLite query.
    - If previous_sql/previous_error are given, the model is asked to fix its earlier mistake (error-retry path).
    - If context is given (list of {"question":.., "sql":..} dicts, most recent last), the model is told about
      the recent conversation so follow-ups like "now break that down by category" resolve correctly."""
    columns = ", ".join(f"{r['name']} ({r['type']})" for _, r in schema.iterrows())

    context_block = ""
    if context:
        turns = "\n".join(f'- Q: "{c["question"]}" -> SQL: {c["sql"]}' for c in context)
        context_block = f"""Recent conversation for context (the new question may refer back to these, e.g. "that", "those", "instead", "now break it down by X"):
{turns}

"""

    if previous_sql and previous_error:
        prompt = f"""You are a SQL expert. Table name: sales. Columns: {columns}.
{context_block}The question is: "{question}"
Your previous SQL attempt failed:
SQL: {previous_sql}
Error: {previous_error}
Write ONE corrected SQLite query that fixes this error and answers the question.
Rules: return ONLY the SQL query, no explanation, no markdown fences."""
    else:
        prompt = f"""You are a SQL expert. Table name: sales. Columns: {columns}.
{context_block}Write ONE SQLite query that answers this question: "{question}"
If the question refers to a previous result (e.g. "that", "those", "now by category"), use the recent conversation above to resolve what it means.
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
def run_query_with_retry(question: str, schema: pd.DataFrame, conn, max_retries: int = 1, context: list = None):
    """Generate SQL, validate it, run it, and retry once with the error fed back to the model if it fails."""
    sql = nl_to_sql(question, schema, context=context)
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

# ---------- FORECASTING ----------
def make_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int, freq: str):
    """Fit a simple linear regression on a time-ordered numeric series and extrapolate forward.
    Also reports MAE on a held-out tail of the historical data so the accuracy is honest, not just assumed."""
    ts = df[[date_col, value_col]].dropna().copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna().sort_values(date_col)
    ts = ts.groupby(date_col, as_index=False)[value_col].sum()

    if len(ts) < 4:
        raise ValueError("Not enough historical data points to forecast (need at least 4).")

    ts["t"] = np.arange(len(ts))
    X = ts[["t"]].values
    y = ts[value_col].values

    # Hold out the last 20% (min 1 point) to honestly evaluate accuracy before trusting the forecast
    split = max(1, int(len(ts) * 0.2))
    X_train, y_train = X[:-split], y[:-split]
    X_test, y_test = X[-split:], y[-split:]

    model = LinearRegression().fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)

    # Refit on all data for the actual forward forecast
    final_model = LinearRegression().fit(X, y)
    future_t = np.arange(len(ts), len(ts) + periods).reshape(-1, 1)
    future_dates = pd.date_range(ts[date_col].iloc[-1], periods=periods + 1, freq=freq)[1:]
    forecast_values = final_model.predict(future_t)

    history = ts[[date_col, value_col]].rename(columns={value_col: "value"})
    history["type"] = "Actual"
    future = pd.DataFrame({date_col: future_dates, "value": forecast_values, "type": "Forecast"})
    combined = pd.concat([history, future], ignore_index=True)
    return combined, mae, future

# ---------- TABS: CHAT + FORECAST ----------
tab_chat, tab_forecast = st.tabs(["💬 Ask a Question", "📈 Forecast"])

with tab_chat:
    # ---------- CHAT UI ----------
    if "history" not in st.session_state:
        st.session_state.history = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    def get_recent_context(n: int = 2):
        """Pull the last n successful Q->SQL exchanges from history so follow-up questions
        (e.g. 'now break that down by category') can be resolved against what was just asked."""
        pairs = []
        h = st.session_state.history
        for i in range(len(h) - 1):
            if h[i][0] == "user" and h[i + 1][0] == "assistant":
                q = h[i][1]
                _, sql, result, error, _ = h[i + 1]
                if sql and error is None:
                    pairs.append({"question": q, "sql": sql})
        return pairs[-n:]

    def process_question(q: str):
        """Runs a question through the pipeline and stores the result in history.
        SQL is kept internally (still generated, still executed) but never shown to the user."""
        context = get_recent_context()
        st.session_state.history.append(("user", q))
        if not client:
            sql, result, error, summary = None, None, "No GROQ_API_KEY found. Add it to .streamlit/secrets.toml to enable the chatbot.", None
        else:
            with st.spinner("Thinking..."):
                sql, result, error = run_query_with_retry(q, schema_df, conn, context=context)
                summary = None
                if result is not None and not result.empty:
                    try:
                        summary = summarize_result(q, result)
                    except Exception:
                        summary = None
        st.session_state.history.append(("assistant", sql, result, error, summary))

    # Suggested prompt buttons — clicking one replaces ONLY that slot with a fresh question,
    # the other three stay exactly as they were.
    SUGGESTION_POOL = [
        "What is the total revenue by region?",
        "What are the top 5 products by sales?",
        "Which category has the highest profit?",
        "Show total sales by month.",
        "Which region has the most orders?",
        "What is the average order value by category?",
        "Which product has the lowest profit margin?",
        "Show total profit by region for this year.",
        "What are the top 3 categories by revenue?",
        "Which month had the highest sales?",
    ]

    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = SUGGESTION_POOL[:4]

    def swap_suggestion(slot_index: int, clicked_question: str):
        """Sends the clicked question to chat, then replaces only that slot with an unused question."""
        st.session_state.pending_question = clicked_question
        used = set(st.session_state.suggested_questions)
        remaining = [q for q in SUGGESTION_POOL if q not in used]
        if remaining:
            st.session_state.suggested_questions[slot_index] = remaining[0]
        # if the pool is exhausted, the slot just keeps its current question

    st.caption("Try one of these, or type your own question below:")
    cols = st.columns(len(st.session_state.suggested_questions))
    for i, (col, sq) in enumerate(zip(cols, st.session_state.suggested_questions)):
        if col.button(sq, use_container_width=True, key=f"suggestion_{i}"):
            swap_suggestion(i, sq)

    question = st.chat_input("e.g. What is the total revenue by region?")

    if question:
        st.session_state.pending_question = question

    if st.session_state.pending_question:
        process_question(st.session_state.pending_question)
        st.session_state.pending_question = None
        st.rerun()

    for entry in st.session_state.history:
        if entry[0] == "user":
            with st.chat_message("user"):
                st.write(entry[1])
        else:
            _, sql, result, error, summary = entry
            with st.chat_message("assistant"):
                if error:
                    st.error(error)
                if result is not None:
                    if summary:
                        st.markdown(f"**Summary:** {summary}")
                    st.dataframe(result, use_container_width=True)
                    numeric_cols = result.select_dtypes("number").columns
                    if len(result.columns) >= 2 and len(numeric_cols) >= 1:
                        label_col = [c for c in result.columns if c not in numeric_cols][0] if len(result.columns) > len(numeric_cols) else result.columns[0]
                        st.bar_chart(result, x=label_col, y=numeric_cols[0])

with tab_forecast:
    st.subheader("Forecast a metric forward in time")

    full_df = pd.read_sql("SELECT * FROM sales", conn)
    date_candidates = [c for c in full_df.columns if "date" in c.lower()]
    numeric_candidates = full_df.select_dtypes("number").columns.tolist()

    if not date_candidates or not numeric_candidates:
        st.warning("No date column and/or numeric column detected in the current dataset — forecasting needs at least one of each.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            date_col = st.selectbox("Date column", date_candidates)
        with col2:
            value_col = st.selectbox("Metric to forecast", numeric_candidates)
        with col3:
            periods = st.number_input("Periods to forecast", min_value=1, max_value=24, value=6)
        with col4:
            freq_label = st.selectbox("Period type", ["Monthly", "Weekly", "Daily"])
            freq = {"Monthly": "MS", "Weekly": "W", "Daily": "D"}[freq_label]

        if st.button("Run forecast"):
            try:
                combined, mae, future = make_forecast(full_df, date_col, value_col, periods, freq)
                st.line_chart(combined, x=date_col, y="value", color="type")
                st.dataframe(future.rename(columns={"value": f"forecasted_{value_col}"}), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Couldn't generate a forecast: {e}")