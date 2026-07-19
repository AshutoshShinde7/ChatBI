import streamlit as st
import pandas as pd
import sqlite3
import re
import os
from groq import Groq

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="ChatBI", page_icon="assets/favicon.ico", layout="wide")
col1, col2 = st.columns([1, 5], vertical_alignment="center")
with col1:
    st.image("assets/chatbi_logo_200.png", width=70)
with col2:
    st.title("ChatBI")
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

def nl_to_sql(question: str, schema: pd.DataFrame) -> str:
    """Ask a Groq-hosted model to turn a natural-language question into a SQLite query."""
    columns = ", ".join(f"{r['name']} ({r['type']})" for _, r in schema.iterrows())
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

# ---------- CHAT UI ----------
if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("e.g. What is the total revenue by region?")

if question:
    st.session_state.history.append(("user", question))
    if not client:
        answer_sql = None
        error = "No GROQ_API_KEY found. Add it to .streamlit/secrets.toml to enable the chatbot."
    else:
        try:
            answer_sql = nl_to_sql(question, schema_df)
            error = None
        except Exception as e:
            answer_sql, error = None, str(e)
    st.session_state.history.append(("assistant_sql", answer_sql, error))

for entry in st.session_state.history:
    if entry[0] == "user":
        with st.chat_message("user"):
            st.write(entry[1])
    else:
        _, sql, error = entry
        with st.chat_message("assistant"):
            if error:
                st.error(error)
            elif sql:
                st.code(sql, language="sql")
                try:
                    result = pd.read_sql(sql, conn)
                    st.dataframe(result, use_container_width=True)
                    numeric_cols = result.select_dtypes("number").columns
                    if len(result.columns) >= 2 and len(numeric_cols) >= 1:
                        label_col = [c for c in result.columns if c not in numeric_cols][0] if len(result.columns) > len(numeric_cols) else result.columns[0]
                        st.bar_chart(result, x=label_col, y=numeric_cols[0])
                except Exception as e:
                    st.error(f"Query failed: {e}")

st.divider()
st.caption("ChatBI — built with Streamlit + Groq API (Llama 3.3) + SQLite. Portfolio project by Ashutosh Shinde.")