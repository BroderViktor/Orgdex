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

os.environ["OPENAI_API_KEY"] = "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"
openai = OpenAI(temperature=0.75)

embedding = OpenAIEmbeddings()

i = input("To Load Existing Index Store, enter 1, to add to existing store enter 2 ")
if (i == "1"):
    db = Chroma(persist_directory="webloader", embedding_function=embedding)

elif (i == '2'):
    db = Chroma(persist_directory="webloader", embedding_function=embedding)
    
    webPaths = []
    url = 0
    while url != "1":
        url = input("Skriv url eller '1' for avslutte")
        if (url == "1"): continue
        webPaths.append(url)

    loader = WebBaseLoader(webPaths)
    documents = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)

    db.add_documents(docs)

else:
    new_path = input("Name for new index store")

    webPaths = []
    url = 0
    while url != "1":
        url = input("Skriv url eller '1' for avslutte")
        if (url == "1"): continue
        webPaths.append(url)
    
    print(webPaths)

    loader = WebBaseLoader(webPaths)
    documents = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    db = Chroma.from_documents(docs, embedding, persist_directory=new_path)

prompt_template = """You are a helpful assistant, informing potentiall customers on the advantages of orgbrain 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use the following pieces of context to answer the question at the end. 

Relevant data: {context}

Previous interactions: {history}

Question: {question}

Answer in the same language that the question was asked in:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "history", "question"]
)

chain = LLMChain(llm=openai, prompt=PROMPT)

memorydata = ConversationBufferWindowMemory(k=3)
while True:
    query = input("Query: ")
    topicdata = db.similarity_search(query, k=2)

    result = chain.run({"context": topicdata, "history": memorydata.chat_memory, "question": query})
    #qa_chain = load_qa_chain(openai, chain_type="stuff")
    #qa = AnalyzeDocumentChain(combine_docs_chain=qa_chain, prompt=PROMPT, verbose=True)
    #qa = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
    #result = qa({"question": query, "chat_history": chat_history})
    #result = qa.run({"input_documents": topicdata, "question": query}, return_only_outputs=True)

    memorydata.chat_memory.add_user_message(query)
    memorydata.chat_memory.add_ai_message(result)

    print(result)

# topicdata = db.similarity_search(query, k=3)

# qa_chain = load_qa_chain(llm, chain_type="map_reduce")
# qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain, verbose=True)

# combinedData = topicdata[0].page_content + topicdata[1].page_content + topicdata[2].page_content

# respons = qa_document_chain.run(input_document=combinedData, question=query)
# print("Respons: " + respons)
