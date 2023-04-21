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

elif (i == '3'):
    db = Chroma(persist_directory="webloader", embedding_function=embedding)

    path = input("Skriv en fil path du vil legge til: ")
    loader = TextLoader(path)
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

prompt_template = """
You are a helpful assistant, informing potentiall customers on the advantages of orgbrain, 
if you don't know the answer to the question, just say that you don't know, don't try to make up an answer, 
Use the following pieces of context to answer the question at the end, provide an full answer that is very relevant to the question
and preferably add a conclusion to the end of the answer. 

Previous interactions: {history}

Relevant data: {context}

Question: {question}

Always answer in the same language that the question was asked in:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "history", "question"]
)

chain = LLMChain(llm=openai, prompt=PROMPT)

memorydata = ConversationBufferWindowMemory(k=3)
with get_openai_callback() as cb:
    while True:

            query = input("Query: ")
            topicdata = db.similarity_search(query, k=3)
            #print(topicdata)
            result = chain.run({"context": topicdata, "history": memorydata.chat_memory, "question": query})
            
            memorydata.chat_memory.add_user_message(query)
            memorydata.chat_memory.add_ai_message(result)

            print(result)
            print(f"Total Tokens: {cb.total_tokens}")
            usdPricing = (cb.total_tokens / 1000) * 0.02
            print(f"Total Cost (NOK): kr {round(usdPricing * 10.59, 4)}")

            #qa_chain = load_qa_chain(openai, chain_type="stuff")
            #qa = AnalyzeDocumentChain(combine_docs_chain=qa_chain, prompt=PROMPT, verbose=True)
            #qa = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
            #result = qa({"question": query, "chat_history": chat_history})
            #result = qa.run({"input_documents": topicdata, "question": query}, return_only_outputs=True)



# topicdata = db.similarity_search(query, k=3)

# qa_chain = load_qa_chain(llm, chain_type="map_reduce")
# qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain, verbose=True)

# combinedData = topicdata[0].page_content + topicdata[1].page_content + topicdata[2].page_content

# respons = qa_document_chain.run(input_document=combinedData, question=query)
# print("Respons: " + respons)
