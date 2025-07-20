from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
import streamlit as st


llm = ChatOllama(model = "llama3.2")
embeddings = OllamaEmbeddings(model = "nomic-embed-text")

def get_docs_from_excel(excel_docs):
    docs = []
    for excel in excel_docs:
        df = pd.read_excel(excel)
        for _, row in df.iterrows():
            content = str(row.get("Message",""))
            docs.append(Document(page_content=content))
    return docs

def embed_docs(docs):
    # langchain_docs = [Document(page_content=doc) for doc in docs]
    return FAISS.from_documents(docs,embeddings )

# def vectorstore(vecstore):
#     return vectorstore.as_retriever()


@tool

def get_ans(query):
    "answer questions about the uploaded excel"
    
    docs = retriever.get_relevant_documents(query, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    
    prompt = f"""You are a data assistant. Use the following context to answer the query.
Context:
{context}

Query: {query}

Insight:"""
    
    return llm.invoke(prompt).content


def setup_system(excel_path):
    global retriever
    docs = get_docs_from_excel([excel_path])
    vs = embed_docs(docs)
    retriever = vs.as_retriever(search_kwargs={"k": 5})


tools = [get_ans]

agent = create_react_agent(llm, tools, name = "qa")

workflow = create_supervisor(
    [agent],
    model=llm,
    prompt = (
        "You are a team supervisor managing data assistant agent and if anyhting is asked use agent for the answer"
    )
)






# app.py


st.set_page_config(page_title="Excel Insight RAG", layout="wide")
st.title("📊 Excel RAG System with LangGraph")

uploaded_file = st.file_uploader("Upload an Excel File (.xlsx)", type=["xlsx"])

if uploaded_file:
    file_path = "temp_uploaded.xlsx"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    setup_system(file_path)
    st.success("✅ Excel file loaded and indexed!")

    query = st.text_input("Ask a question about your data:")
    if st.button("Ask"):
        if query.strip() == "":
            st.warning("Please enter a question.")
        else:
            app = workflow.compile()
            result = app.invoke({"input": query})
            st.markdown("### 🧠 Answer")
            if "final" in result:
                st.markdown("### 🧠 Answer")
                st.write(result["final"])
            else:
                st.warning("⚠️ Could not retrieve a response.")
                st.json(result)



