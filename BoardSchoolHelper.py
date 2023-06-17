import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
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
import re
keys = open("Keys.txt", "r").read().split("\n")
os.environ["OPENAI_API_KEY"] = keys[0]
openai_curie = OpenAI(temperature=0.75)
openai = ChatOpenAI(temperature=0.75, model_name="gpt-3.5-turbo")

embedding = OpenAIEmbeddings()

chunksize = 250
chunkoverlap = 25
kNodes = 4

db = 0


def LoadIndex(inIndex):
    return Chroma(persist_directory=inIndex, embedding_function=embedding)

    return
    webPaths = open(file, "r").read().split("\n")

    loader = WebBaseLoader(webPaths)
    documents = loader.load()
    for doc in documents:
        doc.page_content = doc.page_content.replace("\n", "")

    text_splitter = TokenTextSplitter(chunk_size=chunksize, chunk_overlap=chunkoverlap)
    return text_splitter.split_documents(documents)

def AddFilePath():
    path = input("Skriv en fil path du vil legge til: ")
    loader = TextLoader(path)
    documents = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=chunksize, chunk_overlap=chunkoverlap)
    return text_splitter.split_documents(documents)

def AddFolderPath(path):
    docs = []
    text_splitter = TokenTextSplitter(chunk_size=chunksize, chunk_overlap=chunkoverlap)

    for file in os.listdir(path):
        filePath = (path + "/" + file)
        loader = TextLoader(filePath)
        documents = loader.load()
        docs.extend(text_splitter.split_documents(documents))
    
    for doc in docs:
        sourceName = str(doc.metadata["source"])
        doc.metadata["source"] = "Kurs om " + str(re.split("/|\\.", sourceName)[1])
        
    return docs

#Defines the chain that will answer questions
prompt_template = """
The board school is a course about boards and how to manage them, you are an AI assistant tasked to answer questions concerning the board school
Use the following pieces of context to answer the question at the end, provide an full answer that is very relevant to the question.

Previous interactions, only use this if its relevant to the question: 
{history}

Relevant topic data: 
{context}

Question: 
{question}

If you don't know the answer to the question, just say that you don't know, don't try to make up an answer.
Svar på spørsmålet på norsk
"""
prompt2 = """
The board school is a course about boards and how to manage them, 
you are an AI assistant tasked to answer questions concerning the board school and general questions about board work
Use the following pieces of context to answer the question at the end, provide an full answer that is very relevant to the question.

Previous interactions, only use this if its relevant to the question: 
{history}

You get access to the following documents that are related to the topic: 
{context}

Question: 
{question}

If you don't know the answer to the question, just say that you don't know, don't try to make up an answer.
Svar på spørsmålet på norsk
"""
prompt3 = """
The board school is a course about boards and how to manage them, 
you are an AI assistant tasked to answer questions concerning the board school and general questions about board work
Use the following pieces of context to answer the question at the end, provide an full answer that is very relevant to the question.

Chat history (use only if necessary): 
{history}

Context data (documents from board school): 
{context}

Question: 
{question}

If you don't know the answer to the question, just say that you don't know, don't try to make up an answer.
Svar på spørsmålet på norsk
"""


PROMPT = PromptTemplate(
    template=prompt3, input_variables=["context", "history", "question"]
)
defaultChain = LLMChain(llm=openai, prompt=PROMPT)

db_search_promt = """
Extract notable keywords from the input to search a vector database, limit the amount of words between 1 - 5
Input:
{input}
"""
SEARCH_PROMT = PromptTemplate(
    template=db_search_promt, input_variables=["input"]
)
search_chain = LLMChain(llm=openai_curie, prompt=SEARCH_PROMT)
#Adds memory to the conversation
memorydata = ConversationBufferWindowMemory(k=2)
#Query the llm
def RunQuery(query, index, mem, cb, totalTokenCost, k):

    keywords = search_chain.run({"input": query})
    print("keyword query: " + keywords)
    topicdata = index.similarity_search(keywords, k)
    print(topicdata)

    topicOut = open("topic.txt", "w", encoding="utf-8")
    topicOut.write(str(topicdata))

    result = defaultChain.run({"context": topicdata, "history": mem.load_memory_variables({}), "question": query})
    
    mem.save_context({"User input": query}, {"AI answer": result})

    if (not cb): return result

    usdCost = ((cb.total_tokens - totalTokenCost) / 1000) * 0.002
    print(f"Kostnad for spørsmål (NOK): kr {round(usdCost * 10.59, 4)}")

    usdCost = (cb.total_tokens / 1000) * 0.002
    print(f"Antall tokens i samtale: {cb.total_tokens}")
    print(f"Total kostnad for samtale (NOK): kr {round(usdCost * 10.59, 4)}")

    return result

#Normal QA setup
def QA():
    with get_openai_callback() as cb:
        totalTokenCost = 0
        while True:
            query = input("Query: ")
            answer = RunQuery(query=query, index=db, mem=memorydata, cb=cb, totalTokenCost=totalTokenCost, k=kNodes)
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
            testResult += RunQuery(query=question, index=db, mem=memorydata, cb=cb, totalTokenCost=totalTokenCost, k=kNodes) + "\n\n"
            totalTokenCost = cb.total_tokens
        
        usdcost = (cb.total_tokens / 1000) * 0.02
        cost = f"Antall tokens i samtale: {cb.total_tokens}, dette kostet total (NOK): kr {round(usdcost * 10.59, 4)}"
        testResult += cost
        
        notSaved = True
        saveIndex = 0
        while notSaved:
            saveFile = "Test Results " + str(saveIndex) + " " + index + " nodes " + str(kNodes)
            if (not os.path.exists(saveFile)):
                result = open(saveFile, "w")
                notSaved = False
            saveIndex += 1

        result.write(testResult)


index = "BoardSchool25025"

i = input("To Load Existing Index Store and start QA enter 1, to change indexstore enter 2, to create new enter 3, to test index enter 4")

if (i == "1"):
    db = LoadIndex(index)
    QA()
elif (i == '2'):
    db = LoadIndex(index)
    i = input("Enter 1 to add weburls, 2 to add file path, 3 to add folder")
    if (i == '1'):
        docs = ""# AddUrlPaths()
    elif (i == '2'):
        docs = AddFilePath()
    elif (i == '3'): 
        docs = AddFolderPath(input("path"))
        exit()
    db.add_documents(docs)
elif (i == '3'):
    new_path = input("Name for new index store")
    #docs = ""# AddUrlPathsFromFile("Urls.txt")
    docs = AddFolderPath("Styre Skolen")
    db = Chroma.from_documents(docs, embedding, persist_directory=new_path)
elif (i =='4'):
    db = LoadIndex(index)
    StandardTest()