# # [Upload PDF] → [Extract Text] → [Chunk & Embed] → [Query Input]
# #                     ↑                               ↓
# #                 [Store Vector DB] ← [Retrieve Chunks] → [Answer]


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

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

from typing import TypedDict, Optional, List, Union

VECTORSTORE_DIR = "faiss_index"
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

#vector store
class AgentState(TypedDict):
    filepath: str
    file_hash: str
    docs: List[Document]
    chunks: List[Document]
    question: str
    answer: str
    vectorstore: Optional[FAISS]
    
def get_file_hash(filepath: str) -> str:
    """Generate a hash of the file to detect changes"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def upload_file(state:AgentState):
    current_hash = get_file_hash(state['filepath'])
    
    # Skip processing if file hasn't changed
    if 'file_hash' in state and state['file_hash'] == current_hash:
        print("File unchanged - skipping processing")
        return state
    
    state['file_hash'] = current_hash
    filepath = state['filepath']
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    state['docs'] = pages
    return state


def chunking(state:AgentState):
    # Skip if we already have chunks from unchanged file
    if 'chunks' in state and state['chunks']:
        return state
    
    text_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, )
    chunks = text_split.split_documents(state["docs"])
    state['chunks'] = chunks
    return state
    
def embed(state:AgentState):
    # Skip if we already have a vectorstore
    if 'vectorstore' in state and state['vectorstore'] is not None:
        return state
    embedding = OpenAIEmbeddings(model = "text-embedding-3-small")
    if os.path.exists(VECTORSTORE_DIR):
        print("Loading existing vector store")
        state['vectorstore'] = FAISS.load_local(
            VECTORSTORE_DIR, 
            embedding, 
            allow_dangerous_deserialization=True
        )
    else:
        print("Creating new vector store")
        vectorDB = FAISS.from_documents(state['chunks'], embeddings)
        vectorDB.save_local(VECTORSTORE_DIR)
        state['vectorstore'] = vectorDB
    return state

def query(state:AgentState):
    if 'vectorstore' not in state or state['vectorstore'] is None:
        raise ValueError("Vector store not initialized")
    
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    DB = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    prompt = hub.pull("rlm/rag-prompt")
    retriever = DB.as_retriever()
    model = ChatOpenAI(model = "gpt-4o-mini", max_tokens = 200)
    qa = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    
    answer = qa.run(state['question'])
    state['answer'] = answer
    return state


graph = StateGraph(AgentState)

# Define types of states (data keys)
graph.add_node("upload_pdf", upload_file)
graph.add_node("chunk_", chunking)
graph.add_node("embed", embed)
graph.add_node("query", query)

graph.set_entry_point("upload_pdf")
graph.add_edge("upload_pdf", "chunk_")
graph.add_edge("chunk_", "embed")
graph.add_edge("embed", "query")
graph.add_edge("query", END)

app = graph.compile()

from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

with open("PDF_QNA.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())

# Initialize state
initial_state = AgentState(
    filepath="Problem Statement.pdf",
    docs=[],
    chunks=[],
    question="What is the main topic of the document?",
    answer="",
    vectorstore=None
)

# Execute the graph
final_state = app.invoke(initial_state)
print(final_state["answer"])


# # Streamlit UI
# st.set_page_config(page_title="PDF Q&A Assistant", layout="wide")
# st.title("📄 PDF Question Answering System")

# # File upload section
# uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
# question = st.text_input("Enter your question about the document:")

# if uploaded_file and question:
#     # Save uploaded file temporarily
#     filepath = os.path.join(CACHE_DIR, uploaded_file.name)
#     with open(filepath, "wb") as f:
#         f.write(uploaded_file.getbuffer())
    
#     # Initialize state
#     initial_state = AgentState(
#         filepath=filepath,
#         file_hash="",
#         docs=[],
#         chunks=[],
#         question=question,
#         answer="",
#         vectorstore=None
#     )

#     # Process and display results
#     with st.spinner("Processing your document..."):
#         app = StateGraph(AgentState)
#         final_state = app.invoke(initial_state)
        
#         st.success("Processing complete!")
#         st.subheader("Answer:")
#         st.write(final_state["answer"])

#     # Clean up temporary file
#     os.remove(filepath)

# # Sidebar with info
# with st.sidebar:
#     st.header("About")
#     st.write("""
#     This application lets you upload PDF documents and ask questions about their content.
#     The system will:
#     1. Extract text from your PDF
#     2. Process and chunk the content
#     3. Create searchable embeddings
#     4. Answer your question using AI
#     """)
#     st.markdown("---")
#     st.write("Note: The first time you upload a document, processing may take a while as it creates the search index.")
