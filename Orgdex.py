import os
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import AnalyzeDocumentChain
from langchain.chains import QAGenerationChain
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks import get_openai_callback

os.environ["OPENAI_API_KEY"] = "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"
openai = OpenAI(temperature=0.75)

embedding = OpenAIEmbeddings()

chunksize = 250
chunkoverlap = 25

db = 0


def LoadIndex(inIndex):
    return Chroma(persist_directory=inIndex, embedding_function=embedding)

def AddUrlPaths():
    webPaths = []
    url = 0
    while url != "1":
        url = input("Skriv url eller '1' for avslutte")
        if (url == "1"): continue
        webPaths.append(url)

    loader = WebBaseLoader(webPaths)
    documents = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=chunksize, chunk_overlap=chunkoverlap)
    return text_splitter.split_documents(documents)

def AddUrlPathsFromFile(file):
    webPaths = open(file, "r").read().split("\n")

    loader = WebBaseLoader(webPaths)
    documents = loader.load()

    text_splitter = TokenTextSplitter(chunk_size=chunksize, chunk_overlap=chunkoverlap)
    return text_splitter.split_documents(documents)

def AddFilePath():
    path = input("Skriv en fil path du vil legge til: ")
    loader = TextLoader(path)
    documents = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=chunksize, chunk_overlap=chunkoverlap)
    return text_splitter.split_documents(documents)

#Defines the chain that will answer questions
prompt_template = """
You are a helpful assistant, informing potentiall customers on the advantages of orgbrain, 
Use the following pieces of context to answer the question at the end, provide an full answer that is very relevant to the question.

Previous interactions: 
{history}

Relevant topic data: 
{context}

Question: 
{question}

If you don't know the answer to the question, just say that you don't know, don't try to make up an answer.
If you use any context data from the web, rembember to cite it with a url, if its from another source you do not need to cite it.
Always answer in the same language that the question was asked in:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "history", "question"]
)
chain = LLMChain(llm=openai, prompt=PROMPT)
#Adds memory to the conversation
memorydata = ConversationBufferWindowMemory(k=2)
#Query the llm
def RunQuery(query, index, mem, cb, totalTokenCost):
    topicdata = index.similarity_search(query, k=4)
    
    print(topicdata)
    print(mem.load_memory_variables({}))

    result = chain.run({"context": topicdata, "history": mem.load_memory_variables({}), "question": query})
    
    mem.save_context({"User input": query}, {"AI answer": result})

    if (not cb): return result

    usdCost = ((cb.total_tokens - totalTokenCost) / 1000) * 0.02
    print(f"Kostnad for spørsmål (NOK): kr {round(usdCost * 10.59, 4)}")

    usdCost = (cb.total_tokens / 1000) * 0.02
    print(f"Antall tokens i samtale: {cb.total_tokens}")
    print(f"Total kostnad for samtale (NOK): kr {round(usdCost * 10.59, 4)}")

    return result

#Normal QA setup
def QA():
    with get_openai_callback() as cb:
        totalTokenCost = 0
        while True:
            query = input("Query: ")
            answer = RunQuery(query=query, index=db, mem=memorydata, cb=cb, totalTokenCost=totalTokenCost)
            print(answer)
            totalTokenCost = cb.total_tokens

#Test to check answer quality and costs
def StandardTest():
    with get_openai_callback() as cb:
        totalTokenCost = 0
        questions = open("TestQuestions", "r").read().split("\n")
        testResult = ""
        for question in questions:
            testResult += question + "\n"
            testResult += RunQuery(query=question, index=db, mem=memorydata, cb=cb, totalTokenCost=totalTokenCost) + "\n"
            totalTokenCost = cb.total_tokens
        
        usdcost = (cb.total_tokens / 1000) * 0.02
        cost = f"Antall tokens i samtale: {cb.total_tokens}, dette kostet total (NOK): kr {round(usdcost * 10.59, 4)}"
        testResult += cost
        result = open("testresultater " + index, "w")
        result.write(testResult)


index = "StandardIndex 50 5"

i = input("To Load Existing Index Store and start QA enter 1, to change indexstore enter 2, to create new enter 3, to test index enter 4")

if (i == "1"):
    db = LoadIndex(index)
    QA()
elif (i == '2'):
    db = LoadIndex(index)
    if (input("Enter 1 to add weburls") == '1'):
        docs = AddUrlPaths()
    else:
        docs = AddFilePath()
    
    db.add_documents(docs)
elif (i == '3'):
    new_path = input("Name for new index store")
    docs = AddUrlPathsFromFile("Urls.txt")
    db = Chroma.from_documents(docs, embedding, persist_directory=new_path)
elif (i =='4'):
    db = LoadIndex(index)
    StandardTest()