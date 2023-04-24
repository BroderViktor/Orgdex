from langchain.agents import Tool, AgentExecutor
from langchain.agents.agent import AgentOutputParser, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import BaseTool
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import re
import os

os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] = "b7a188d3356b5839e4702806655a1c7eef82b141f3e4e6f75bb21977c0a9b70a"
llm = OpenAI(temperature=0)
embedding = OpenAIEmbeddings()

index = Chroma(persist_directory="chunksize100", embedding_function=embedding)

# Define which tools the agent can use to answer user queries
search = SerpAPIWrapper()

class IndexSearch(BaseTool):
    name = "Search Orgbrain database"
    description = "useful for when you need to answer questions about orgbrain"

    def _run(self, query: str) -> str:
        """Use the tool."""
        return "Orgbrain er et selskap som jobber med selskaper for å øke produktiviteten av styrearbeidet, noen annsatte er dag asheim og thomas evensen, de har over 1000 kunder"
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("does not support async")

def searchIndex(query):
    return index.similarity_search(query, k=4)

tools = [
    Tool(
        name = "Search memory",
        func = searchIndex,
        description = "Get specific information about orgbrain"
    ),
    Tool(
        name = "Search web",
        func=search.run,
        description="Get general information"
    ),
]

# Set up the base template
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}
"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):

    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)
output_parser = CustomOutputParser()

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
with get_openai_callback() as cb:
    agent_executor.run("hva er orgbrain?")
    usdCost = ((cb.total_tokens) / 1000) * 0.02
    print(f"Kostnad for spørsmål (NOK): kr {round(usdCost * 10.59, 4)}")
