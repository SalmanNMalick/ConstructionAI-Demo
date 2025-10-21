# Section 1: Design and Build AI Agent Frameworks
# Demonstrating autonomous agents using LangChain and Hugging Face for LLMs.
# This example builds a simple multi-agent system for construction task automation, e.g., knowledge retrieval and project management assistance.
# Note: In a real environment, install langchain, langchain-community, transformers via pip if not available.
# Also, set HUGGINGFACEHUB_API_TOKEN environment variable.

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory

# Define tools for agents (e.g., knowledge retrieval from construction docs, data interpretation)
def retrieve_construction_knowledge(query):
    """Simulates retrieving knowledge from construction datasets."""
    knowledge_base = {
        "foundation types": "Slab, crawl space, basement - choose based on soil and climate.",
        "project timeline": "Planning: 2 weeks, Construction: 3 months, Inspection: 1 week."
    }
    return knowledge_base.get(query.lower(), "No information found.")

knowledge_tool = Tool(
    name="ConstructionKnowledgeRetrieval",
    func=retrieve_construction_knowledge,
    description="Retrieve information from construction knowledge base."
)

def interpret_data(data):
    """Interprets unstructured construction data, e.g., site reports."""
    # Simple parsing example
    if "delay" in data.lower():
        return "Project delay detected due to weather; suggest rescheduling."
    return "Data interpreted as normal."

data_tool = Tool(
    name="DataInterpretation",
    func=interpret_data,
    description="Interpret construction site data."
)

# Set up LLM (using Hugging Face)
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5})

# Prompt for ReAct agent
prompt = PromptTemplate.from_template(
    "You are a construction AI agent. Use tools to answer: {input}\n{agent_scratchpad}"
)

# Create agent
agent = create_react_agent(llm, [knowledge_tool, data_tool], prompt)
memory = ConversationBufferMemory()
agent_executor = AgentExecutor(agent=agent, tools=[knowledge_tool, data_tool], memory=memory, verbose=True)

# Example usage: Automate task
response = agent_executor.invoke({"input": "What foundation type for sandy soil? And interpret this site report: Heavy rain causing delay."})
print(response['output'])
