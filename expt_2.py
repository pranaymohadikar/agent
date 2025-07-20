# from dotenv import load_dotenv
# import os
# import pandas as pd
# from langchain_community.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
# from langchain_ollama import ChatOllama, OllamaEmbeddings
# from langchain_core.tools import tool
# from langgraph.prebuilt import create_react_agent
# from langgraph_supervisor import create_supervisor
# from langgraph.graph import END, StateGraph
# from typing import Annotated, Sequence, TypedDict
# from operator import add as add_messages
# from langchain_core.messages import BaseMessage, HumanMessage
# from langchain_community.vectorstores import FAISS
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain.chains.question_answering import load_qa_chain

# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", default=None)

# model = ChatOpenAI(model = "gpt-4o-mini", temperature=0,)
# embedding_1 = OpenAIEmbeddings(model = "text-embedding-3-small")

# # === Step 1: Load Excel File ===
# excel_path = "temp_uploaded.xlsx"
# if not os.path.exists(excel_path):
#     raise FileNotFoundError(f"Excel file not found: {excel_path}")

# df = pd.read_excel(excel_path)
# documents = []

# for _, row in df.iterrows():
#     content = str(row.get("Message"))
#     #print(content)
#     metadata={
#     "Sentiment": row.get("Sentiment"),
#     "Date": str(row.get("Created On (Posted On)")),
#     "Source": row.get("Message Source"),}
#     documents.append(Document(page_content=content, metadata = metadata))
# #print(documents[10])
# print(f"Loaded {len(documents)} rows from Excel.")

# # === Step 2: Chunking ===
# # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# # chunks = text_splitter.split_documents(documents)

# # === Step 3: Embedding and VectorStore ===
# embeddings = OllamaEmbeddings(model="nomic-embed-text")
# persist_directory = r"E:\KaaM\LangGraph\LangGraph_tut"
# collection_name = "test_excel"

# if not os.path.exists(persist_directory):
#     os.makedirs(persist_directory)

# # vectorstore = Chroma.from_documents(
# #     documents=documents,
# #     embedding=embeddings,
# #     persist_directory=persist_directory,
# #     collection_name=collection_name
# # )

# faiss_index_path = os.path.join(persist_directory, "faiss_index")

# # Check if FAISS index already exists
# if os.path.exists(faiss_index_path):
#     print("Loading FAISS index...")
#     vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
# else:
#     print("Creating FAISS index...")
#     vectorstore = FAISS.from_documents(documents, embeddings)
#     vectorstore.save_local(faiss_index_path)

# retriever = vectorstore.as_retriever(search_type="similarity")

# # === Step 4: Tool ===
# llm = ChatOllama(model="llama3.2", temperature=0)
# # retrieval_qa_chain = RetrievalQA.from_chain_type(
# #     llm = model,
# #     retriever = retriever,
# #     chain_type = "stuff"
# # )

# # def retriever_tool(query: str) -> str:
# #     """Search for info from the Stock Market Excel file."""
# #     docs = retriever.invoke(query)
# #     if not docs:
# #         return "No relevant information found in the Excel document."

# #     results = [f"Chunk {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
# #     return "\n\n".join(results)




# # def qa_tool(query: str):
    
# #     """Answer questions using RetrievalQA on Excel data."""
    
    
# #     result = retrieval_qa_chain.run(query)
# #     return result
# # === Step 5: LangGraph Agent Setup ===

# prompt_template = """
# Answer the question as detailed as possible from the provided context. and if someone ask you to summarize sentiment wise, use the sentiment provided in the data. dont assume sentiment itself.
#     If the answer is not available in the context, say "Answer is not available in the context."
    
#     Context: {context}
#     Question: {question}
    
#     Answer:
#     """

# qa_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# # Load QA chain
# qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)


# def qa_tool(query: str) -> str:
#     """Use vector search + QA chain to answer questions from Excel data."""
#     docs = retriever.invoke(query)
#     if not docs:
#         return "No relevant information found in the Excel document."
    
#     result = qa_chain.run(input_documents=docs, question=query)
#     return result


# tools = [qa_tool]

# # Use LangChain's create_react_agent for the agent logic
# react_agent = create_react_agent(model =llm, tools=tools, 
#                                  name = "qa",
#                                  prompt="""
# You are a helpful AI assistant that answers user questions using data stored in an Excel file.
# The file contains **user messages** related to feedbacks, along with **metadata** 

# You MUST always use the provided `qa_tool` to search for relevant content before responding.
# Your role is not just to extract information, but also to summarize it clearly and concisely **based on the retrieved results**.

# ### Instructions:
# - Use the `qa_tool` to fetch relevant chunks.
# - Then, based on the retrieved information, **generate a final answer**.
# - If the user asks for a **summary**, provide a concise summary using the retrieved data.
# - If no relevant chunks are found, respond with: "No relevant information found in the Excel document."
# - You must NOT guess or generate answers without using the retriever first.
# - Your task is to summarize the key themes, topics, and sentiments expressed provide me a consice summary of the messages.



# Always use the chunks retrieved from the qa tool to inform your response.
# """
# )

# # === Step 6: Define Agent State ===
# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]

# # === Step 7: Create Supervisor to handle looping logic ===
# supervisor = create_supervisor([react_agent],
#                                model = llm,
#                                prompt="""
# You are the supervisor managing a team of agents.

# Whenever the user asks a question related to feedback, sentiment, or summaries from the Excel file, delegate it to the `qa` agent.

# The `qa` agent will use a specialized `qa_tool` to fetch relevant messages and generate a well-informed answer.

# Do not attempt to answer the question yourself — only route the query to the appropriate agent.
# """)

# # === Step 8: LangGraph setup ===

# rag_agent = supervisor.compile()
# # === Step 9: Run ===
# def running_agent():
#     print("\n=== RAG AGENT (Excel + LangGraph) ===")
#     while True:
#         user_input = input("\nWhat is your question: ")
#         if user_input.lower() in ['exit', 'quit', "e"]:
#             break
#         messages = [HumanMessage(content=user_input)]
#         result = rag_agent.invoke({"messages": messages})
#         print("\n=== ANSWER ===")
#         print(result['messages'][-1].content)


# running_agent()

import dateparser

question = "Get messages count on 19th may"
parsed = dateparser.parse(question)
print("🧪 Raw parse result:", parsed)