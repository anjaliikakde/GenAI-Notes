from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
from .llm import get_llm
from .tools import get_tools

def setup_research_assistant():
    """Setup and return the research assistant agent executor."""
    llm = get_llm()
    tools = get_tools()
    
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True)