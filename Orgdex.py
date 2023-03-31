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
from langchain.chains import ConversationChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain

os.environ["OPENAI_API_KEY"] = "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"
llm = OpenAI(temperature=0.75)

embedding = OpenAIEmbeddings()

i = input("To Load Existing Index Store, enter 1")
if (i == "1"):
    db = Chroma(persist_directory="webloader", embedding_function=embedding)
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

query = input("Query: ")
DataAboutTopic = db.similarity_search(query, k=1)
print(DataAboutTopic)

qa_chain = load_qa_chain(llm, chain_type="map_reduce")
qa_document_chain = ConversationChain()

combinedData = DataAboutTopic[0].page_content + DataAboutTopic[1].page_content

respons = qa_document_chain.run(input_document=combinedData, question=query)
print("Respons: " + respons)
