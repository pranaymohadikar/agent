from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from typing import TypedDict


class AgentState(TypedDict):
    title: str
    outline: str
    content: str
    
model = ChatOllama(model = "llama3.2")
    
def create_outline(state: AgentState):
    title = state['title']
    prompt = f'Generate  a detailed outline for the blog on the topic - {title}'
    outline = model.invoke(prompt).content
    
    state['outline'] = outline
    
    return state


def create_blog(state: AgentState):
    title = state['title']
    outline = state['outline']
    prompt = f'write a detailed blog on the title- {title} using following outline {outline}'
    
    content = model.invoke(prompt).content
    state['content'] = content
    
    return state


graph = StateGraph(AgentState)
graph.add_node("create outline", create_outline)
graph.add_node("create blog", create_blog)

graph.add_edge(START, "create outline")
graph.add_edge("create outline", "create blog")
graph.add_edge("create blog", END)


workflow = graph.compile()


initial_state = {"title":"rise of agents" }
final = workflow.invoke(initial_state)

print(final["content"])