import pandas as pd
import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langgraph.graph import END
from typing import List

# ---------------------------------------------
# Tools
# ---------------------------------------------

@tool
def get_documents_from_csv(csv_docs: List[str]) -> List[Document]:
    """Load messages and metadata from uploaded CSV files."""
    documents = []
    for csv in csv_docs:
        df = pd.read_csv(csv)
        for _, row in df.iterrows():
            content = str(row.get("Message", ""))
            metadata = {
                "date": row.get("Created On (Posted On)", "Unknown"),
                "location": row.get("Administrative Area", "Unknown"),
                "sentiment": row.get("Sentiment", "Unknown")
            }
            documents.append(Document(page_content=content, metadata=metadata))
    return documents

@tool
def build_vector_store(documents: List[Document]) -> str:
    """Create FAISS vector store from documents."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return "Vector store created"

@tool
def summarize_peak_dates(csv_docs: List[str]) -> str:
    """Summarize peak message dates from CSV files."""
    all_data = pd.concat([pd.read_csv(csv) for csv in csv_docs], ignore_index=True)
    date_col = "Created On (Posted On)"
    if date_col not in all_data.columns or "Message" not in all_data.columns:
        return "Required columns missing."
    date_counts = all_data[date_col].value_counts().nlargest(1)

    results = []
    for date, count in date_counts.items():
        messages = all_data[all_data[date_col] == date]["Message"].dropna().astype(str).tolist()
        documents = [Document(page_content=msg) for msg in messages]

        prompt = PromptTemplate(
            template="""
            You are given a list of user messages from a specific date.
            Your task is to summarize the key themes, topics, and sentiments expressed.

            Messages:
            {context}

            Summary:
            """,
            input_variables=["context"]
        )
        chain = load_qa_chain(ChatOllama(model="llama3.2"), chain_type="stuff", prompt=prompt)
        summary = chain({"input_documents": documents}, return_only_outputs=True)["output_text"]

        results.append(f"Date: {date}\nCount: {count}\nSummary: {summary}")

    return "\n---\n".join(results)

@tool
def ask_question(query: str) -> str:
    """Ask a question from FAISS vectorstore."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query, k=10)

    prompt = PromptTemplate(
        template="""
        Answer the question as detailed as possible from the provided context.
        If the answer is not available in the context, say "Answer is not available in the context."

        Context: {context}
        Question: {question}

        Answer:
        """,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(ChatOllama(model="llama3.2"), chain_type="stuff", prompt=prompt)
    return chain({"input_documents": docs, "question": query}, return_only_outputs=True)["output_text"]

# ---------------------------------------------
# LangGraph Agent & Supervisor
# ---------------------------------------------
tools = [get_documents_from_csv, build_vector_store, summarize_peak_dates, ask_question]
agent = create_react_agent(ChatOllama(model="llama3.2"), tools, name = "agent")
supervisor = create_supervisor([agent], model = ChatOllama(model="llama3.2", prompt ="" ))

# Define the state
class AgentState():
    question: str
    result: str = ""

def run_agent_pipeline(question: str):
    app = supervisor.compile()
    config = {"configurable": {"thread_id": "xyz"}}
    inputs = {"question": question}
    result = app.invoke(inputs, config=config)
    final = list(result)[-1]
    return final

# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------
def main():
    st.set_page_config(page_title="LangGraph CSV Chat", layout="wide")
    st.title("Chat with CSV using LangGraph")

    uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)
    if st.button("Process Files"):
        if uploaded_files:
            csv_paths = [f.name for f in uploaded_files]  # simulate file paths
            doc_output = get_documents_from_csv.invoke({"csv_docs": csv_paths})
            st.success("Documents loaded.")
            vector_output = build_vector_store.invoke({"documents": doc_output})
            st.success(vector_output)

    question = st.text_input("Ask your question")
    if question:
        answer = run_agent_pipeline(question)
        st.markdown(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()
