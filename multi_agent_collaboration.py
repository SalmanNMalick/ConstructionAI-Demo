# Section 3: Research and Implement Advanced Methods in Multi-Agent Collaboration
# Using AutoGen for multi-agent systems with planning and reasoning.
# Agents collaborate on construction planning (e.g., one plans, another reasons on risks).
# Note: Requires OpenAI API key in OAI_CONFIG_LIST file or env.

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# Load config (assume OpenAI or Hugging Face API keys)
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

# Define agents
user_proxy = UserProxyAgent(
    name="User",
    system_message="A human user simulating construction manager.",
    code_execution_config={"work_dir": "planning"},
)

planner_agent = AssistantAgent(
    name="Planner",
    system_message="You plan construction tasks step-by-step.",
    llm_config={"config_list": config_list},
)

reasoner_agent = AssistantAgent(
    name="Reasoner",
    system_message="You reason on risks and optimizations for plans.",
    llm_config={"config_list": config_list},
)

# Group chat for collaboration
from autogen import GroupChat, GroupChatManager

groupchat = GroupChat(agents=[user_proxy, planner_agent, reasoner_agent], messages=[], max_round=10)
manager = GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

# Initiate collaboration
user_proxy.initiate_chat(
    manager,
    message="Plan a foundation build for a 3-story building and reason on weather risks."
)
