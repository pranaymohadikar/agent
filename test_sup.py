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
import os
from typing import Annotated, Sequence, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage


# === Step 1: Load Excel File ===
excel_path = "temp_uploaded.xlsx"
if not os.path.exists(excel_path):
    raise FileNotFoundError(f"Excel file not found: {excel_path}")

df = pd.read_excel(excel_path)
documents = []

for _, row in df.iterrows():
    content = str(row.get("Message"))
    #print(content)
    metadata={
    "Sentiment": row.get("Sentiment"),
    "Date": str(row.get("Created On (Posted On)")),
    "Source": row.get("Message Source"),}
    documents.append(Document(page_content=content, metadata = metadata))
#print(documents[10])
print(f"Loaded {len(documents)} rows from Excel.")



# === Step 3: Embedding and VectorStore ===
embeddings = OllamaEmbeddings(model="nomic-embed-text")
persist_directory = r"E:\KaaM\LangGraph\LangGraph_tut"
#collection_name = "test_excel"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)
    
faiss_index_path = os.path.join(persist_directory, "faiss_index")

# Check if FAISS index already exists
if os.path.exists(faiss_index_path):
    print("Loading FAISS index...")
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
else:
    print("Creating FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(faiss_index_path)

retriever = vectorstore.as_retriever(search_type="similarity")

# === Step 4: Tool ===
llm = ChatOllama(model="llama3.2", temperature=0)


prompt_template = """
Answer the question as detailed as possible from the provided context. and if someone ask you to summarize sentiment wise, use the sentiment provided in the data. dont assume sentiment itself.
    If the answer is not available in the context, say "Answer is not available in the context."
    
    Context: {context}
    Question: {question}
    
    Answer:
    """

qa_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# Load QA chain
qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)


def qa_tool(query: str) -> str:
    """Use vector search + QA chain to answer questions from Excel data."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the Excel document."
    
    result = qa_chain.run(input_documents=docs, question=query)
    return result


general_agent = create_react_agent(model =llm, tools=[qa_tool], 
                                 name = "qa",
                                 prompt="""
You are a helpful AI assistant that answers user questions using data stored in an csv file.
The file contains **user messages** related to feedbacks, along with **metadata** 

You MUST always use the provided `qa_tool` to search for relevant content before responding.
Your role is not just to extract information, but also to summarize it clearly and concisely **based on the retrieved results**.

### Instructions:
- Use the `qa_tool` to fetch relevant chunks.
- Then, based on the retrieved information, **generate a final answer**.
- If the user asks for a **summary**, provide a concise summary using the retrieved data.
- If no relevant chunks are found, respond with: "No relevant information found in the Excel document."
- You must NOT guess or generate answers without using the retriever first.

Always use the chunks retrieved from the qa tool to inform your response.
"""
)

def summarize_peak_dates(csv_docs: List[str]) -> dict:
    """Summarize peak message dates from CSV files."""
    all_data = pd.concat([pd.read_csv(csv) for csv in csv_docs], ignore_index=True)
    date_col = "Created On (Posted On)"
    if date_col not in all_data.columns or "Message" not in all_data.columns:
        return {"result": "Required columns missing."}
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

    return {"result": "\n---\n".join(results)}

peak_date_summary = create_react_agent(model = llm, tools = [summarize_peak_dates], name = 'peak date summary',
                                       prompt = "you are summary peak agent whose work is if someone ask about the peak date of the data, you assist it then only")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    
workflow = create_supervisor(agents=[general_agent, peak_date_summary], model = llm,
                             prompt = "you are the supervisor managing team of agent. when the generic question is asked like summarize the conversations delegate it to 'qa_tool' or if someone asked about the peak dates, delegate it to 'summarize_peak_dates'. Do not attempt to answer the question yourself — only route the query to the appropriate agent. ")


app = workflow.compile()

def running_agent():
    print("\n=== RAG AGENT (Excel + LangGraph) ===")
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit', "e"]:
            break
        messages = [HumanMessage(content=user_input)]
        result = app.invoke({"messages": messages})
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()






