from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sqlite3
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
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
from new import graph, schema


app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class AgentState(TypedDict):
    question: str
    parsed_date: str
    query: str
    result: str
    
    
@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        input_state = {"question": request.question}
        output = graph.invoke(input_state)
        
        return {
            "question": request.question,
            "sql_query": output["query"],
            "result": output["result"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schema")
async def get_schema():
    return {"schema": schema}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)