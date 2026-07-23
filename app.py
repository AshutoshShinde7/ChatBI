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

# ---------- PERSONA (used so identity questions like "who made you" get a real, consistent answer) ----------
CHATBI_PERSONA = """You are ChatBI, a natural-language data analyst assistant. Facts about you, if asked:
- You were built by Ashutosh Shinde, a Data Science graduate, as a portfolio project.
- You're built with Streamlit for the interface, the Groq API running Llama models for language understanding, and SQLite for querying the data.
- You can answer questions about the user's uploaded dataset, and forecast a metric forward in time (Forecast tab).
Only mention these facts if actually asked who made you, what you're built with, or what you can do — don't volunteer this in every reply."""

# ---------- CONVERSATION ROUTER (handles greetings/small talk vs. real data questions) ----------
def route_message(message: str, transcript: str = ""):
    """Decides if this is small talk (greeting, thanks, identity questions, etc.) or an actual
    data question. Small talk gets a natural, direct reply — it never hits the SQL pipeline.
    Uses the small/fast model since this is a simple classification+reply task, not query generation."""
    transcript_block = f"\nRecent conversation so far:\n{transcript}\n" if transcript else ""

    prompt = f"""{CHATBI_PERSONA}

You're texting back and forth with someone. You're sharp and easygoing, like a coworker who's good with numbers, not a customer service bot.
{transcript_block}
They just said: "{message}"

First, decide: are they asking an actual data question (numbers, totals, comparisons, trends, records about their dataset) or are they just talking to you (greeting, thanks, small talk, or a question about you — like who made you or what you can do)?

If it's a DATA QUESTION, reply with exactly: DATA_QUESTION

If it's NOT a data question, reply like a real person texting back — don't reuse a stock phrase, actually respond to what they said. Rules:
- Use contractions (I'm, you're, that's, don't)
- Keep it short, usually one sentence
- No corporate phrases: never say "I'm here to help", "feel free to", "I'd be happy to", "let me know if you have any questions"
- If they ask who made you, what you're built with, or what you can do, answer using the facts above — briefly, not a full readout
- Pay attention to the recent conversation above so you don't repeat something you already said
- Never say the exact same thing twice in a row"""

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=120,
        temperature=1.0,
        messages=[{"role": "user", "content": prompt}],
    )
    reply = resp.choices[0].message.content.strip()
    if reply.upper().startswith("DATA_QUESTION"):
        return None  # signals: proceed to the SQL pipeline
    return reply

# ---------- RESULT SUMMARY (uses a smaller/faster model — cheaper for a simple task) ----------
def summarize_result(question: str, df: pd.DataFrame) -> str:
    preview = df.head(20).to_csv(index=False)
    prompt = f"""You're ChatBI, texting someone the answer to a question about their data. They asked: "{question}"

Here's what the query returned (CSV, up to 20 rows):
{preview}

Tell them what it shows, like you're glancing at the numbers and casually pointing out what matters — 1-3 sentences. Rules:
- Use contractions and everyday words
- Lead with the actual number or finding, not a restatement of their question
- No corporate/report phrasing — avoid "the data indicates", "it can be observed that", "in summary", "overall"
- Don't repeat the question back to them
- If something stands out (a big gap, a surprising leader, a clear trend), react to it naturally instead of just listing facts
- Vary how you open — don't always start with the same word or phrase"""
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=150,
        temperature=1.0,
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

# ---------- CHAT STATE + HELPERS (top-level, so chat_input below can pin to the page bottom) ----------
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

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

def get_raw_transcript(n_turns: int = 4) -> str:
    """Builds a plain-text version of the last n_turns exchanges (both data questions AND
    small talk), so the router actually remembers what was just said instead of only
    knowing about SQL history. This is what fixes repeated/generic small-talk replies."""
    h = st.session_state.history
    lines = []
    for i in range(len(h) - 1):
        if h[i][0] == "user" and h[i + 1][0] == "assistant":
            user_msg = h[i][1]
            _, sql, result, error, summary = h[i + 1]
            if summary:
                bot_msg = summary
            elif error:
                bot_msg = "(ran into an error)"
            elif result is not None:
                bot_msg = "(showed a data table)"
            else:
                bot_msg = ""
            lines.append(f"User: {user_msg}")
            lines.append(f"You: {bot_msg}")
    return "\n".join(lines[-(n_turns * 2):])

def process_question(q: str):
    """Runs a question through the pipeline and stores the result in history.
    SQL is kept internally (still generated, still executed) but never shown to the user.
    Small talk / greetings are detected first and answered directly, without touching SQL at all."""
    context = get_recent_context()
    transcript = get_raw_transcript()
    st.session_state.history.append(("user", q))
    if not client:
        sql, result, error, summary = None, None, "No GROQ_API_KEY found. Add it to .streamlit/secrets.toml to enable the chatbot.", None
    else:
        with st.spinner("Thinking..."):
            chat_reply = route_message(q, transcript=transcript)
            if chat_reply is not None:
                # Small talk — just a conversational reply, no SQL, no table, no chart
                sql, result, error, summary = None, None, None, chat_reply
            else:
                sql, result, error = run_query_with_retry(q, schema_df, conn, context=context)
                summary = None
                if result is not None and not result.empty:
                    try:
                        summary = summarize_result(q, result)
                    except Exception:
                        summary = None
    st.session_state.history.append(("assistant", sql, result, error, summary))

def swap_suggestion(slot_index: int, clicked_question: str):
    """Sends the clicked question to chat, then replaces ONLY that slot with an unused question
    (rebuilds the list explicitly rather than mutating in place, to avoid any stale-reference issues)."""
    st.session_state.pending_question = clicked_question
    used = set(st.session_state.suggested_questions)
    remaining = [q for q in SUGGESTION_POOL if q not in used]
    new_list = list(st.session_state.suggested_questions)
    if remaining:
        new_list[slot_index] = remaining[0]
    st.session_state.suggested_questions = new_list

# ---------- TABS: CHAT + FORECAST ----------
tab_chat, tab_forecast = st.tabs(["💬 Ask a Question", "📈 Forecast"])

with tab_chat:
    if not st.session_state.history:
        with st.chat_message("assistant"):
            st.write("Hey! Ask me anything about your data — I'll dig up the numbers. There's also a Forecast tab if you want to project something forward.")
    for entry in st.session_state.history:
        if entry[0] == "user":
            with st.chat_message("user"):
                st.write(entry[1])
        else:
            _, sql, result, error, summary = entry
            with st.chat_message("assistant"):
                if error:
                    st.error(error)
                elif result is not None:
                    if summary:
                        st.write(summary)
                    st.dataframe(result, use_container_width=True)
                    numeric_cols = result.select_dtypes("number").columns
                    if len(result.columns) >= 2 and len(numeric_cols) >= 1:
                        label_col = [c for c in result.columns if c not in numeric_cols][0] if len(result.columns) > len(numeric_cols) else result.columns[0]
                        st.bar_chart(result, x=label_col, y=numeric_cols[0])
                elif summary:
                    # small talk / conversational reply — no data attached
                    st.write(summary)

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

# ---------- SUGGESTED PROMPTS + CHAT INPUT (both top-level, so they sit together at the bottom of the page) ----------
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