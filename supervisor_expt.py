from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", default=None)

model = ChatOpenAI(model = "gpt-4o-mini", temperature=0,)


def add(a,b):
    "add 2 numbers"
    return a+b


def sub(a,b):
    "subtract 2 numbers"
    return a-b


def mul(a,b):
    "multiply 2 numbers"
    return a*b


def div(a,b):
    "divide 2 numbers"
    return a/b



def search(query:str):
    "search from web"
    search = DuckDuckGoSearchRun()
    return search.invoke(query)
    
#creating agents

math_agent = create_react_agent(
    model = model,
    tools = [add, mul, sub, div],
    name = "math expert",
    prompt = "you are an math expert. always use one tool at a time")


research_agent = create_react_agent(
    model = model,
    tools = [search],
    name = "research_expert",
    prompt="You are a world class researcher with access to web search. Do not do any math."
)

#create a supervisor workflow

workflow = create_supervisor(
    [math_agent, research_agent],
    model=model,
    prompt = (
        "You are a team supervisor managing a rresearch expert and math expert"
        "For current events use research_agent"
        "for math problem, use math_agent"
    )
)

# Compile and run
app = workflow.compile()
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "what is AI"
        }
    ]
})

print(result)