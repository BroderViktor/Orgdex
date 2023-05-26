from langchain.agents import Tool, AgentExecutor
from langchain.agents.agent import AgentOutputParser, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from abc import ABC
from langchain.schema import AgentAction, AgentFinish, BaseMemory, BaseChatMessageHistory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.tools import BaseTool
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.utilities import WikipediaAPIWrapper
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
from langchain.memory.utils import get_prompt_input_key
from pydantic import Field
import re
import os
from serpapi import GoogleSearch

keys = open("Keys.txt", "r").read().split("\n")
os.environ["OPENAI_API_KEY"] = keys[0]
os.environ["GOOGLE_CSE_ID"] = keys[1]
os.environ["GOOGLE_API_KEY"] = keys[2]
os.environ["SERPAPI_API_KEY"] = keys[3]
cse_id = keys[1]
api_key = keys[2]

llm = OpenAI(temperature=0)
embedding = OpenAIEmbeddings()

index = Chroma(persist_directory="FormatedDocs25025", embedding_function=embedding)

# Define which tools the agent can use to answer user queries

from googleapiclient.discovery import build

search = SerpAPIWrapper(params={ "engine": "google","google_domain": "google.com","gl": "no","hl": "no" })
wikipedia_search = WikipediaAPIWrapper(top_k_results=2)


def google_search(search_term, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res['items']

def google_search_parser(search, kNum = 3):
    search_results = google_search(search, num=kNum)
    parsed_results = []
    for res in search_results:
        #print(res)
        parsed_results.append(res["title"] + ": " + res["snippet"])

    return format(parsed_results)

def searchIndex(query):
    documents = index.similarity_search(query, k=3)

    for document in documents:
        print(document)
        if ("title" in document.metadata):
            tempSourcesMemory[0].append("title: " + document.metadata["title"] + " Source: " + document.metadata["source"])
        else:
            tempSourcesMemory[0].append("Data fra " + document.metadata["source"] + " kommer fra lokalt minne")

    return documents

def searchGoogleWrap(query):

    search = GoogleSearch({
        "q": query, 
        "hl": "no",
        "gl": "no",
        "location": "Oslo,Norway",
        "api_key": keys[3],
    })
    #open("test.json", "w").write(format(search.get_json()))
    search_res = search.get_dict()
    results = search_res["organic_results"][:2]
    parsedResults = []

    keys_to_delete = []
    answerbox = ""
    if ("answer_box" in search_res):
        answerbox = search_res["answer_box"]
        for key, value in answerbox.items():
            if (not isinstance(value, str)):
                keys_to_delete.append(key)
                
    for key in keys_to_delete:
        del answerbox[key]

    if (answerbox != ""):
        parsedResults.append("Data: " + format(answerbox))
        tempSourcesMemory[0].append("Google search: " + search_res["search_metadata"]["google_url"])

    for res in results:
        parsedResults.append("According to: [" + res["title"] + "] their website claims: [" + res["snippet"] + "] link: " + res["link"])
        tempSourcesMemory[0].append("Source for: [" + res["title"] + "] " + res["link"])

    return parsedResults

def checkWebsite(query):
    return format(tempSourcesMemory)

def searchSources(query):
    return format(tempSourcesMemory)

tools=[]

tools.extend([
    Tool(
        name = "Search Database",
        func = searchIndex,
        description = "Get specific information about orgbrain or other business related questions, this should be your primary source, input should be a norwegian question"
    ),
    Tool(
        name = "Google Search",
        func = searchGoogleWrap,
        description = "A wrapper around Google Search. Useful for when you need to answer questions about current events. Input should be a search query."
    ),
    Tool(
        name = "Conversation Memory",
        func = searchSources,
        description = "Look up sources that have been used in this conversation previously, no input to be given"
    ),
    Tool(
        name = "Wikipedia Search",
        func = wikipedia_search.run,
        description = "Get information on a large number of topics, input should be a search query in english"
    ),
])

# Set up the base template
template = """You are an Agent for customers of orgbrain.no, answer the following questions as best you can giving as much information as is relevant. 
You have access to the following tools:

{tools}

There are no other tools available.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 2 times)

Thought: I now know the final answer
Final Answer: the final answer to the original input question

{chat_history}

Question (always answer the question in the same language as this): {input}

{agent_scratchpad}
"""

class AdvancedMemory(ConversationBufferWindowMemory):

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str)

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
            print(f"Could not parse LLM output: `{llm_output}`")

        action = match.group(1).strip()
        action_input = match.group(2)

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps", "chat_history"]
)
memory = AdvancedMemory(memory_key="chat_history", k=2)
tempSourcesMemory = []

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

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
with get_openai_callback() as cb:
    active = True
    while active:
        tempSourcesMemory.insert(0, [])
        agent_executor.run(input("Question: "))
        if (len(tempSourcesMemory) > 1): tempSourcesMemory.pop()
        usdCost = ((cb.total_tokens) / 1000) * 0.02
        print(f"Kostnad for spørsmål (NOK): kr {round(usdCost * 10.59, 4)}")

    #agent_executor.run("hva er orgbrain?")
    #print(memory.chat_memory)

