#!/usr/bin/env python
# coding: utf-8

# ## https://en.wikipedia.org/wiki/Virat_Kohli
# ## simple gen ai app with langchain 
# 

import os
from dotenv import load_dotenv
from  langchain_openai import ChatOpenAI

## load the enviroment variables 
load_dotenv();
# // this is for the  model of the openai 
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
#  this is for the langsmith tracking 
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
#  lanchain trackeing 
os.environ["LANGCHAIN_TRACING_V2"] = "true";
#  this is the project name for the tracking of the langsmith 
os.environ["LANCHAIN_PROJECT_NAME"] = os.getenv("LANCHAIN_PROJECT_NAME")
print("loadding complete")

llm = ChatOpenAI(model ="gpt-4o")
print(llm)

## data injection -- from the website we have to scrape the data 
from langchain_community.document_loaders import WebBaseLoader

url = "https://en.wikipedia.org/wiki/Virat_Kohli"
loder = WebBaseLoader(url)
docs = loder.load()
for doc in docs:
    doc.page_content = " ".join(doc.page_content.split())

## load data --> docs -->divide our data into chunks -->text splitter --> create embeddings --> store in vector database --> retrive the data from the vector database --> use llm to generate the answer to the question

from  langchain_text_splitters  import RecursiveCharacterTextSplitter
# split the documents into chunks of 1000 characters with an overlap of 200 characters
 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

text_splitter.split_documents(documents= docs)

# // now use the embedding techinique to converst the tect to vectors

from langchain_ollama import  OllamaEmbeddings
embedding = OllamaEmbeddings(
   model = "nomic-embed-text:latest"
)

from langchain_community.vectorstores import  FAISS
vectorstoresdb = FAISS.from_documents(docs,embedding)

query = "hey how much runs virat scores in the ipl ?"
result = vectorstoresdb.similarity_search(query);
result[0].page_content

## retrivatal chain , document chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

promt = ChatPromptTemplate.from_template(
    '''
    answer the question answer based on the provided context :
    <context>
    </context>
    {context}
    '''
)
document_chain = create_stuff_documents_chain(
    llm,promt
    
)

print(document_chain)

from langchain_core.documents import  Document
document_chain.invoke({
    "input":"user giveing the input",
    "context":[Document(page_content="user is the land")]
})

# however we want the documents to first come from the retrivers we just setup that way we can use the retriver to dynamic select the most relevernt documents and pass those in for a given questuion

# retriever  -- > it can be contside be as a interface and anyonee pass an input this retriver is the way  probbebly geting the data from the vectorstore db

#input -- > retriver --> vectorstoredb

retriver = vectorstoresdb.as_retriever()

from langchain_classic.chains import  create_retrieval_chain
retriver_chain = create_retrieval_chain(retriver,document_chain)

retriver_chain

## get the responces from the llm
responcee = retriver_chain.invoke({"input":"userinput"})
responcee["answer"]

responcee
responce['context']
