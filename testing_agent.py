from langchain_ollama import ChatOllama, OllamaEmbeddings
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.tools import Tool
from langchain.docstore.document import Document
from langchain.tools.retriever import create_retriever_tool
import os
from langgraph.graph import MessagesState
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode





from typing import TypedDict

file = "temp_uploaded.xlsx"
df = pd.read_excel(file)
document = []

for _, row in df.iterrows():
    content = str(row.get("Message"))
    
    document.append(Document(page_content=content))
    
#print(document[0])
 
 
embeddings = OllamaEmbeddings(model = "nomic-embed-text")
llm = ChatOllama(model = "llama3.2")
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
    vectorstore = FAISS.from_documents(document, embeddings)
    vectorstore.save_local(faiss_index_path)
    
    
retriever = vectorstore.as_retriever(search_type="similarity")

retriever_tool = create_retriever_tool( retriever,"ret_tool",
"search and return infofrmation about what the query is asked by user")


answer = retriever_tool.invoke({'query': "what are the  negative conversations"})

#print(answer)

    
class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages] 
    
    
def generate_respond(state):
    messages = state["messages"]
    llm_wt_tool = llm.bind_tools([retriever_tool])
    response = llm_wt_tool.invoke(messages)
    return {"messages": [response]}


def should_continue(state) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


workflow = StateGraph(AgentState)


workflow.add_node("agent", generate_respond)

tool_node = ToolNode([retriever_tool])
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
)
workflow.add_edge("tools", "agent")

graph = workflow.compile()



# from IPython.display import Image, display

# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass

# with open("Agent_2.png", "wb") as f:
#     f.write(graph.get_graph().draw_mermaid_png())
    
from langchain_core.messages import SystemMessage

system_prompt = SystemMessage(content="You are a helpful assistant. Use any retrieved content to summarize the user query in a concise way.")

   
answer = graph.invoke(input={
    "messages": [system_prompt, HumanMessage(content="what is the count of messages on 1st september")][-1]
})

from langchain_core.messages import AIMessage

ai_messages = [m for m in answer["messages"] if isinstance(m, AIMessage)]
final_summary = ai_messages[-1].content
print(final_summary)


 

