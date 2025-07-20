### Directory Structure:
# - app.py (Streamlit UI)
# - rag_core.py (RAG + LangGraph logic)
# - data/messages.csv

# -------------------- rag_core.py --------------------
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings

embedding_model = OllamaEmbeddings(model = "nomic-embed-text")
llm = ChatOllama(model = "llama3.2", temperature=0)

# Load and preprocess 
def get_documents_from_excel(excel_docs):
    documents = []
    for excel in excel_docs:
        df = pd.read_excel(excel)
        for _, row in df.iterrows():
            content = str(row.get("Message", ""))
            documents.append(Document(page_content=content))
    return documents

# Embed and store in vector DB
def embed_docs(docs):
    # splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    langchain_docs = [Document(page_content=doc) for doc in docs]
    return FAISS.from_documents(langchain_docs, embedding_model)

# Retrieve relevant docs from query
def get_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def retrieve_docs(state):
    docs = retriever.invoke(state["input"])
    return {"retrieved_docs": docs, "input": state["input"]}

def tool_call(state):
    query = state["input"].lower()
    docs = state.get("retrieved_docs", [])
    text = "\n".join([doc.page_content for doc in docs])

    if "summary" in query:
        return {"tool_results": f"SUMMARY:\n{text[:1000]}..."}  # replace with actual summary model
    elif "sentiment" in query:
        return {"tool_results": "Approx. 70% positive, 20% neutral, 10% negative."}
    return {"tool_results": None}

def generate_answer(state):
    context = state.get("tool_results") or "\n".join([doc.page_content for doc in state.get("retrieved_docs", [])])
    prompt = f"Answer the question based on the data:\n\nContext:\n{context}\n\nQuestion: {state['input']}"
    response = llm.invoke(prompt)
    return {"final_answer": response.content}

def update_memory(state):
    return state

def build_graph():
    builder = StateGraph()
    builder.add_node("retriever", RunnableLambda(retrieve_docs))
    builder.add_node("tool_caller", RunnableLambda(tool_call))
    builder.add_node("llm_responder", RunnableLambda(generate_answer))
    builder.add_node("memory_updater", RunnableLambda(update_memory))

    builder.set_entry_point("retriever")
    builder.add_edge("retriever", "tool_caller")
    builder.add_edge("tool_caller", "llm_responder")
    builder.add_edge("llm_responder", "memory_updater")
    builder.add_edge("memory_updater", END)
    return builder.compile()



#=============================================================
# builder = StateGraph(state)
# builder.add_node("retriever", RunnableLambda(retrieve_docs))
# builder.add_node("tool_caller", RunnableLambda(tool_call))
# builder.add_node("llm_responder", RunnableLambda(generate_answer))
# builder.add_node("memory_updater", RunnableLambda(update_memory))

# builder.set_entry_point("retriever")
# builder.add_edge("retriever", "tool_caller")
# builder.add_edge("tool_caller", "llm_responder")
# builder.add_edge("llm_responder", "memory_updater")
# builder.add_edge("memory_updater", END)
# app = builder.compile()

# from IPython.display import Image, display

# try:
#     display(Image(app.get_graph().draw_mermaid_png()))
# except Exception:
# # This requires some extra dependencies and is optional
#     pass

# with open("RAG_QNA.png", "wb") as f:
#     f.write(app.get_graph().draw_mermaid_png())












#=========================================================



# Entry point to initialize vector DB and graph
def init_rag_system(uploaded_file):
    global retriever
    df = get_documents_from_excel(uploaded_file)
    vectorstore = embed_docs(df)
    retriever = get_retriever(vectorstore)
    return build_graph()

# -------------------- app.py --------------------
import streamlit as st
#from rag_core import init_rag_system

st.set_page_config(page_title="RAG Chatbot for excel", layout="wide")
st.title("📊 excel Chatbot - LangGraph + RAG")

uploaded_file = st.file_uploader("Upload your excel file", type=["xlsx"])

if uploaded_file:
    with st.spinner("Processing uploaded excel..."):
        st.session_state.rag_graph = init_rag_system(uploaded_file)
        st.session_state.history = []

if "rag_graph" in st.session_state:
    query = st.text_input("Ask something about the excel data:", key="query_input")

    if st.button("Submit") and query:
        with st.spinner("Thinking..."):
            result = st.session_state.rag_graph.invoke({"input": query})
            st.session_state.history.append((query, result["final_answer"]))

    st.write("---")
    st.subheader("Conversation History")
    for user_q, bot_a in reversed(st.session_state.history):
        st.markdown(f"**You:** {user_q}")
        st.markdown(f"**Bot:** {bot_a}")
else:
    st.info("Please upload a excel file to begin.")
