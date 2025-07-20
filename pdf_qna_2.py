from langgraph.graph import StateGraph, START, END
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain import hub
import hashlib
import streamlit as st
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage

VECTORSTORE_DIR = "faiss_index"
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

from typing import TypedDict, Optional, List, Union, Annotated


model = ChatOpenAI(model = "gpt-4o-mini", temperature = 0, max_tokens = 200)

embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

path = "Problem Statement.pdf"

if not os.path.exists(path):
    raise FileNotFoundError(f'file not found')

loader = PyPDFLoader(path)
try:
    pages = loader.load()
    print(len(pages))
except Exception as e:
    print('error in loading file: {e}')
    

#chunking

text_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_split.split_documents(pages)

try:
    vectorDB = FAISS.from_documents(chunks, embeddings)
    FAISS.save_local(VECTORSTORE_DIR)
    print("created")
    
except Exception as e:
    print(f"error {e}")


#retriever
DB = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever =vectorDB.as_retriever(search_type="similarity",
    search_kwargs={"k": 5} # K is the amount of chunks to return)
)

@tool

def retriever_tool(query: str):
    "this tool searched and returns the information from the pdf"
    
    docs = retriever.invoke(query)
    
    if not docs:
        return "no relevent docs found"
    
    result = []
    for i, doc in enumerate(docs):
        result.append('Doc {i+1}: {doc.page_content}')
        
    return "\n".join(result)


tools = [retriever_tool]

model_new = model.bind_tools(tools)

#typeddict model

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    
    
def should_continue(state: AgentState):
    "check f the last message contains tool calls"
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls)>0


system_prompt = """
You are an intelligent AI assistant who answers questions about the PDF you have been provided
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools} # creatingn dict of out tools

#llm agent

def call_llm(state:AgentState):
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)]+messages
    message = model_new.invoke(messages)
    return state


#retriever agent

def take_action(state: AgentState):
    #eecute tol call from llm resposne
    
    tool_calls = state["messages"][-1].too_calls
    result = []
    for i in tool_calls:
        print(f"calling tool: {i['name']} with query: {t['args'].get('query', 'No query provided' )}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

if __name__ == "__main__":
    
    running_agent()


