import os
from langchain.llms import OpenAI
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.base import Chain

from typing import Dict, List

os.environ["OPENAI_API_KEY"] = "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"

class ConcatenateChain(Chain):
    chain_1: LLMChain
    chain_2: LLMChain

    @property
    def input_keys(self) -> List[str]:
        # Union of the input keys of the two chains.
        all_input_vars = set(self.chain_1.input_keys).union(set(self.chain_2.input_keys))
        return list(all_input_vars)

    @property
    def output_keys(self) -> List[str]:
        return ['concat_output']

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        output_1 = self.chain_1.run(inputs)
        output_2 = self.chain_2.run(inputs)
        return {'concat_output': output_1 + output_2}
    

llm = OpenAI(temperature=0.9)
prompt_name = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?, make it cool and interesting",
)
prompt_catchphrase = PromptTemplate(
    input_variables=["product"],
    template="Write a catchphrase for the a company that makes: {product}, short and sweet",
)

name = LLMChain(llm=llm, prompt=prompt_name)
catchphrase = LLMChain(llm=llm, prompt=prompt_catchphrase)
overall_chain = ConcatenateChain(chain_1=name, chain_2=catchphrase, verbose=True)


print(overall_chain.run("military weapons: serious"))