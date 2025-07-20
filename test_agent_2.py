# # trying this timem by not using vectorstore and embeddings


# import pandas as pd
# from datetime import datetime
# from langchain_ollama import ChatOllama
# from langchain_core.tools import tool
# from langgraph.prebuilt import create_react_agent
# from langgraph_supervisor import create_supervisor
# from langgraph.graph import StateGraph, END
# from langchain.prompts import PromptTemplate

# import os
# import re
# import io

# #llm

# llm = ChatOllama(model = "llama3.2")

# #read_csv

# file = "temp_uploaded.xlsx"
# df = pd.read_excel(file)

# df['Created On (Posted On)'] = pd.to_datetime(df['Created On (Posted On)'])


# #tools

# @tool

# #filtering data


# def filter_data(query: str) -> str:
#     '''use llm o generate pandas query to filter the dataframe based on query'''
#     filter_prompt = PromptTemplate.from_template("""
#                                                  You are a python expert working with the dataframe named 'df'. The dataframe has the collowing columns: {columns}.
#                                                  Generate a python code using pandas that flter 'df' based on the following query "{query}"
#                                                  The result should ne assigned to a variable named 'result'. Only return tye code.
#                                                  """
        
#     )
    
#     prompt_input = filter_prompt.format(columns=list(df.columns), query=query)
#     code = llm.predict(prompt_input).strip()

#     # Basic safeguard: Check if it tries to do anything other than df operations
#     if not re.match(r"^result\s*=\s*df[\s\S]*", code):
#         return f"Blocked potentially unsafe code: {code}"

#     try:
#         local_vars = {"df": df.copy(), "datetime": datetime}
#         exec(code, {}, local_vars)
#         result_df = local_vars.get("result")
#         if result_df is not None and isinstance(result_df, pd.DataFrame):
#             return result_df.to_csv(index=False)
#         else:
#             return f"No valid DataFrame returned.\nCode: {code}"
#     except Exception as e:
#         return f"Error executing generated code: {e}\nCode: {code}"
    
    
    
# @tool

# def summarize_data(excel_data: str) -> str:
#     """Summarize the given filtered CSV content."""
#     try:
#         data_df = pd.read_csv(io.StringIO(excel_data))
#         summary = f"Total messages: {len(data_df)}\n"
#         summary += f"Unique sources: {data_df['message source'].nunique()}\n"
#         summary += f"Sample messages: {data_df['messages'].head(3).tolist()}"
#         return summary
#     except Exception as e:
#         return f"Error summarizing data: {e}"
    
    
# # ==== Agents ====
# filter_agent = create_react_agent(llm, tools=[filter_data], name="FilterAgent")
# summarizer_agent = create_react_agent(llm, tools=[summarize_data], name="SummarizerAgent")

# # ==== LangGraph State ====
# class AgentState(dict):
#     query: str = ""
#     result: str = ""

# # ==== Define LangGraph ==== #
# graph = StateGraph(AgentState)

# graph.add_node("filter", filter_agent)
# graph.add_node("summarize", summarizer_agent)

# # Supervisor decides routing
# supervisor = create_supervisor(model = llm, agents = [filter_agent, summarizer_agent], prompt= """
#                                you are a supervisor agent which will delegate the task according to the user query. 
#                                Delegate filterwise task to filter_agent
#                                delegate summary task to summarizer_agent""")

# workflow = supervisor.compile()

# # graph.add_node("supervisor", supervisor)

# # graph.set_entry_point("supervisor")
# # graph.add_edge("supervisor", "filter")
# # graph.add_edge("filter", "summarize")
# # graph.add_edge("summarize", END)

# # # ==== Compile the graph ====
# # app = graph.compile()

# from IPython.display import display, Image

# graph_bytes = workflow.get_graph().draw_mermaid_png()
# with open("graph.png", "wb") as f:
#     f.write(graph_bytes)

# print("Graph saved as 'graph.png'")

# # ==== Run It ====
# if __name__ == "__main__":
#     while True:
#         user_query = input("Ask a question about your excel data (or type 'exit'): ")
#         if user_query.lower() == "exit":
#             break
#         response = workflow.invoke({"query": user_query})
#         print("\nFinal Result:\n", response.get("result"), "\n")
    



import pandas as pd
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from typing import TypedDict
from IPython.display import display, Image
import io

# Initialize LLM
llm = ChatOllama(model="llama3.2")

# Load Data
df = pd.read_excel("temp_uploaded.xlsx")
df['Created On (Posted On)'] = pd.to_datetime(df['Created On (Posted On)'])

# Tools
@tool
def filter_data(query: str) -> str:
    """Filter dataframe using pandas."""
    prompt = f"""
    Generate a pandas query to filter `df` (columns: {list(df.columns)}) based on: "{query}".
    Return only the code, assigning the result to `result`.
    Example: "result = df[df['column'] == 'value']"
    """
    code = llm.invoke(prompt).content.strip()
    
    # Security check
    if not code.startswith("result = df"):
        return f"Blocked unsafe code: {code}"
    
    try:
        local_vars = {"df": df.copy(), "datetime": datetime}
        exec(code, {}, local_vars)
        result_df = local_vars["result"]
        return result_df.to_csv(index=False)
    except Exception as e:
        return f"Error: {e}"

@tool
def summarize_data(csv_data: str) -> str:
    """Summarize CSV data."""
    try:
        data_df = pd.read_csv(io.StringIO(csv_data))
        summary = f"Rows: {len(data_df)}\n"
        summary += f"Columns: {list(data_df.columns)}\n"
        return summary
    except Exception as e:
        return f"Error: {e}"

# Define State
class AgentState(TypedDict):
    query: str
    result: str

# Build Graph
graph = StateGraph(AgentState)
graph.add_node("filter", filter_data)
graph.add_node("summarize", summarize_data)
graph.add_edge("filter", "summarize")
graph.add_edge("summarize", END)
graph.set_entry_point("filter")
app = graph.compile()

# Visualize
graph_bytes = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_bytes)
display(Image(graph_bytes))  # Show in notebook

# Run
if __name__ == "__main__":
    while True:
        user_query = input("Query (or 'exit'): ")
        if user_query.lower() == "exit":
            break
        result = app.invoke({"query": user_query})
        print(result["result"])