from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sqlite3
#from langchain.agents import create_sql_agent
#from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
#from langchain_community.utilities.sql_database import SQLDatabase
from langchain_ollama import ChatOllama
import os
from typing import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
import re
import dateparser
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import uvicorn
from datetime import datetime

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


app = FastAPI()

#llm = ChatOllama(model="llama3.2")
llm = ChatOpenAI(model = "gpt-4o-mini")
file_path = 'fake_messages_dataset.xlsx'
db_path = "fake_db"
table_name = "data"
# query = "select * from data"

df = pd.read_excel(file_path)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
df.to_sql(table_name, conn, if_exists= "replace", index=False)
conn.commit()



cursor.execute(f"PRAGMA table_info({table_name});")

# cols = cursor.fetchall()
# # for row in rows:
# #     print(row)

# #print(cols[1])
# for col in cols:
#     print(f"Column name: {col[1]}")
    
    
df = pd.read_sql("SELECT * FROM data LIMIT 1", conn)
#print(df.head())
conn.close()

def get_schema_from_db(db_path: str, table_name: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    conn.close()

    schema = "\n".join(f"{col[1]} ({col[2]})" for col in columns)
    print(schema)
    return schema


def init_cache_db():
    "initialize cache database if it doesn't exist"
    conn = sqlite3.connect('query_cache.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_cache (
                   normalized_sql TEXT PRIMARY KEY,
                   raw_question TEXT,
                   sql_query TEXT,
                   result TEXT,
                   summary TEXT
                   );
""")
    
    conn.commit()
    conn.close()

init_cache_db()

class AgentState(TypedDict):
    question: str
    parsed_date: str
    query: str
    result: str
    summary: str
    
class QueryRequest(BaseModel):
    question: str



#cache utility fuction

# def normalize_query(q: str):
#     return re.sub(r'\s+', " ", q.strip().lower()) 

def normalize_sql(sql: str) -> str:
    return re.sub(r'\s+', ' ', sql.strip().lower())

# def get_cached_result(normalized_query: str):
#     norm_quest = normalize_query(normalized_query)
#     conn = sqlite3.connect("query_cache.db")
#     cursor = conn.cursor()
#     cursor.execute("""select sql_query, result, summary from query_cache where normalized_query = ?;""", (norm_quest,))

#     row = cursor.fetchone()
#     conn.close()

#     if row:
#         return{
#             "query": row[0],
#             "result": row[1],
#             "summary": row[2]

#         } 
#     return None


def get_cached_result_by_sql(sql_query: str):
    norm_sql = normalize_sql(sql_query)
    conn = sqlite3.connect("query_cache.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT result, summary FROM query_cache
        WHERE normalized_sql = ?;
    """, (norm_sql,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"result": row[0], "summary": row[1]}
    return None

# def store_cache_result(normalized_query: str, sql_query:str, result: str, summary:str):
#     norm_quest = normalize_query(normalized_query)
#     conn = sqlite3.connect("query_cache.db")
#     cursor = conn.cursor()
#     cursor.execute("""
#         INSERT OR REPLACE INTO query_cache (normalized_query, raw_question, sql_query, result, summary)
#         VALUES (?, ?, ?, ?, ?);
#     """, (norm_quest, normalized_query, sql_query, result, summary))
    
#     conn.commit()
#     conn.close()


def store_sql_cache( sql_query: str, question: str, result: str, summary: str):
    norm_sql = normalize_sql(sql_query)
    conn = sqlite3.connect("query_cache.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO query_cache (normalized_sql, raw_question, sql_query, result, summary)
        VALUES (?, ?, ?, ?, ?);
    """, (norm_sql, question, sql_query, result, summary))
    conn.commit()
    conn.close()



schema = get_schema_from_db("fake_db.sqlite", "data")

def route(state: AgentState):
    question = state["question"]

    date_keywords = [
        r"\b(yesterday|today|last|next|week|month|day)\b",  # Combined as regex
        r"\b\d{4}-\d{2}-\d{2}\b",  # YYYY-MM-DD
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",  # MM/DD/YYYY or DD-MM-YYYY
        r"\b\w+ \d{1,2}(st|nd|rd|th)?\b", # "January 5th"
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?\b", # for 1may 19 or th
        r"\b\d{1,2}(?:st|nd|rd|th)? (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b", # ofr 10 may 
    ]

    
    if any(re.search(pattern, question) for pattern in date_keywords):
        return {**state, "next_node":"parse_date", "date_type": "exact"}
    # else: #if no specific date then consider te current month and year
    #     today = datetime.now() 
    #     current_month_year = today.strftime("%Y-%m")
    #     return {**state, "parsed_date": current_month_year, "next_node": "generate_sql", "date_type": "month_year" }


    # Else: get latest year and month from the database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(Date) FROM data")  # Replace with correct column name if needed
        latest_date_str = cursor.fetchone()[0]
        conn.close()

        if latest_date_str:
            latest_date = datetime.strptime(latest_date_str.split()[0], "%Y-%m-%d")
            parsed_date = latest_date.strftime("%Y-%m")
        else:
            parsed_date = datetime.now().strftime("%Y-%m")  # fallback

    except Exception as e:
        print(f"Failed to retrieve latest date from DB: {e}")
        parsed_date = datetime.now().strftime("%Y-%m")

    return {**state, "parsed_date": parsed_date, "next_node": "generate_sql", "date_type": "month_year"}
    
date_patterns = [
    r"\b(yesterday|today|tomorrow|last|next|week|month|day)\b",                       # natural language
    r"\b\d{4}-\d{2}-\d{2}\b",                                                         # YYYY-MM-DD
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",                                             # 19/05/2024 or 05-19-2024
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(st|nd|rd|th)?\b",  # May 19th
    r"\b\d{1,2}(st|nd|rd|th)? (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b",  # 19th May
    r"\b\w+ \d{1,2}(st|nd|rd|th)?\b",                                                 # January 5th
]


def extract_date_like_string(text: str):
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group()
    return None

# def parse_date(state: AgentState):
#     print("parse_date() function was called")
    
#     question = state["question"]
#     parsed = dateparser.parse(question)

#     if not parsed:
#         return {**state, "parsed_date": None}
    
#     '''
#     **state spreads (unpacks) the entire dictionary.

# You're adding or overwriting parsed_date cleanly.

# This is a standard pattern when working with stateful flows like LangGraph.
#     '''
    
#     parsed_date = parsed.strftime("%Y-%m-%d")
#     #print(f"parsed date: {parsed_date}") 
#     return {**state, "parsed_date": parsed_date}

def parse_date(state: AgentState):
    print("parse_date() function was called")
    question = state["question"]

    # Try extracting date-like substring
    
    date_str = extract_date_like_string(question)
    #match = re.search(r"\b\d{1,2}(st|nd|rd|th)? (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*", question, re.IGNORECASE)
    
    if not date_str:
        print("No date-like string found")
        return {**state, "parsed_date": None}

    print(f"Extracted date string: {date_str}")
    parsed = dateparser.parse(date_str)

    if not parsed:
        print("dateparser failed to parse")
        return {**state, "parsed_date": None}

    parsed_date = parsed.strftime("%Y-%m-%d")
    print(f"Parsed date: {parsed_date}")
    return {**state, "parsed_date": parsed_date}


sql_prompt = PromptTemplate.from_template(
    """
You are a SQL expert. Given a question and table schema, convert the question into a valid SQL query.

Use this table schema:
{schema}

Table name is `data`.

Instructions:
- For exact dates: Use `DATE(Date) = '{parsed_date}'` when a specific date is provided
- For general queries without specific dates, filter by current month using strftime
- For partial dates (parsed_date in format YYYY-MM): Use `strftime('%Y-%m', Date) = '{parsed_date}'`
- Use `date = '{parsed_date}'` **if parsed_date is provided**, even if the question contains fuzzy dates like "19th May".
- Always use column names from the schema.
- All string comparisons must be **case-insensitive** using `LOWER(column) = 'value'`.
- For date columns, use `column directly means dont lowercase the column name of date. use Date
- The column containing dates is called `Date` and stores values in the format `YYYY-MM-DD HH:MM:SS`.
- When filtering by date, always use `DATE(Date) = 'YYYY-MM-DD'` to match the date part only (ignore time).
- Be case-sensitive with column names (e.g., use `Date` not `date`.
- If a parsed date is available (e.g., `2025-05-19`), use it in the format `'YYYY-MM-DD'`
- When writing SQL involving dates (e.g., for grouping, filtering, or comparisons), always extract the date part using DATE(Date).
- use specific columns whcih are mentioned in the question, if not mentioned then use all columns.
- Never use `SELECT *`. Instead, select only the relevant columns (e.g., `message`, `location`, etc.).

Here are some examples:

Q: Get all messages with positive sentiment  
SQL: SELECT message FROM data WHERE LOWER(sentiment) = 'positive';

Q: List all locations where sentiment is negative  
SQL: SELECT location FROM data WHERE LOWER(sentiment) = 'negative';

Q: What are the messages from Mumbai?  
SQL: SELECT message FROM data WHERE LOWER(location) = 'mumbai';

Q: Get message count on 19th may or may 10 
Parsed Date: 2025-05-19 
SQL: SELECT COUNT(message) FROM data WHERE DATE(Date) = '2025-05-19';

Q: Get positive messages (no specific date). Here the parsed date is latest year and month whcih you will get from the database
Parsed Date: 2025-07
SQL: SELECT message FROM data WHERE LOWER(sentiment) = 'positive' AND strftime('%Y-%m', Date) = '2025-07';

 


Now answer the following:

Q: {question}  
parsed date : {parsed_date}
date type: {date_type}
SQL: 

"""
)

sql_generator_chain: Runnable = sql_prompt | llm | StrOutputParser()

def llm_generate_sql(state: AgentState):
    question = state["question"]
    parsed_date = state.get("parsed_date", "")
    date_type = state.get("date_type", "")
    

    sql = sql_generator_chain.invoke({
        "question": question,
        "parsed_date": parsed_date,
        "date_type": date_type,
        "schema": schema
    })
         # Clean up the SQL output
    sql = sql.strip()
    # if sql.lower().startswith("sql:"):
    #     sql = sql[4:].strip()
    sql = sql.split(";")[0] + ";" if ";" not in sql[-1] else sql

    return {**state, "query": sql}


def summarizer (state: AgentState):
    question = state["question"]
    query = state['query']
    #parsed_date = state['parsed_date']
    result = state['result']

    summary_prompt = PromptTemplate.from_template(
        """ You are a SQL expert. Given the following question, SQL query, and the result, summarize the information concisely and clearly.

Question:
{question}

Parsed Date:
{parsed_date}

SQL Query:
{query}

SQL Result:
{result}

Summary:
"""
        )
    summary_chain = summary_prompt |llm | StrOutputParser()

    summary = summary_chain.invoke({
        "question": question,
        "parsed_date": state.get("parsed_date", ""),
        "query": query,
        "result": result
    })
    return {**state, "summary": summary}    



def run_sql(state: AgentState):
    sql_query = state.get("query", "").strip()

    
    if sql_query.lower().startswith("```sql:"):
        sql_query = sql_query[7:].strip()
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(sql_query, conn)
        result = df.to_string(index=False) if not df.empty else "no result"
    except Exception as e:
        result = f"sql execution error {e}"
    finally:
        conn.close()
        
    return {**state,"result": result}


builder = StateGraph(AgentState)

builder.add_node("route", route)
builder.add_node("parse_date", parse_date)
builder.add_node("generate_sql", llm_generate_sql)
builder.add_node("run_sql", run_sql)
builder.add_node('summarizer',summarizer)


builder.set_entry_point("route")

# builder.add_conditional_edges(
#     "route",
#     {
#         "parse_date": "parse_date",
#         "generate_sql": "generate_sql"
#     }
# )
# builder.add_edge("route", "parse_date")
# builder.add_edge("route", "generte_sql")

# builder.add_conditional_edges(
#     "route",
#     lambda state: "parse_date" if "parse_date" in route(state) else "generate_sql",
#     {"parse_date": "parse_date", "generate_sql": "generate_sql"}
# )

builder.add_conditional_edges(
    "route",
    lambda state: state["next_node"],
    {"parse_date": "parse_date", "generate_sql": "generate_sql"}
)
builder.add_edge("parse_date", "generate_sql")
builder.add_edge("generate_sql", "run_sql")
builder.add_edge("run_sql", "summarizer")
builder.add_edge('summarizer', END)


graph = builder.compile()


# from IPython.display import display, Image

# graph_bytes = graph.get_graph().draw_mermaid_png()
# with open("SQL.png", "wb") as f:
#     f.write(graph_bytes)

# print("Graph saved as 'SQL.png'")


# input_state = {"question": "provide me summary of negative messages"}
# output = graph.invoke(input_state)

# print("Generated SQL:\n", output["query"])


# #print("\nQuery Result:\n", output["result"]) 
# print("\n Summary: \n ", output['summary'])

question_text = "get me the count of positive messages"
# cached = get_cached_result(question_text)

# if cached:
#     print("Using cached result!")
#     output = {
#         "query": cached["query"],
#         "result": cached["result"],
#         "summary": cached["summary"]
#     }
# else:
#     print("No cache hit. Invoking LangGraph...")
#     input_state = {"question": question_text}
#     output = graph.invoke(input_state)

#     # Store result in cache
#     store_cache_result(
#         question_text,
#         output["query"],
#         output["result"],
#         output["summary"]
#     )

# print("Generated SQL:\n", output["query"])
# print("\nSummary:\n", output["summary"])

input_state = {"question": question_text}
intermediate_output = graph.invoke(input_state)

sql_query = intermediate_output["query"]
cached = get_cached_result_by_sql(sql_query)

if cached:
    print("Using cached result for identical SQL!")
    output = {
        "query": sql_query,
        "result": cached["result"],
        "summary": cached["summary"]
    }
else:
    # Run full flow (SQL execution, summarization)
    print("No cache hit. Running SQL execution and summarization...")
    output = graph.invoke(input_state)
    store_sql_cache(sql_query, question_text, output["result"], output["summary"])


print("Generated SQL:\n", output["query"])
print("\nSummary:\n", output["summary"])



# @app.post("/query")
# async def process_query(request: QueryRequest):
#     try:
#         input_state = {"question": request.question}
#         output = graph.invoke(input_state)
        
#         return {
#             "question": request.question,
#             "sql_query": output["query"],
#             "result": output["result"],
#             "summary": output['summary']
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/schema")
# async def get_schema():
#     new_schema = get_schema_from_db("fake_db.sqlite", "data")
#     return {"schema": new_schema}




# @app.get("/")
# async def root():
#     return {
#         "status": "API is running",
#         "endpoints": {
#             "query": "/query",
#             "schema": "/schema"
#         },
#         "documentation": "/docs"
#     }

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)