#!/usr/bin/env python
# coding: utf-8

# #### Embedding Techniques
# 
# Converting text into vectors

import os
from dotenv import load_dotenv
load_dotenv()  #load all the environment variables

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

from langchain_openai import OpenAIEmbeddings
embeddings=OpenAIEmbeddings(model="text-embedding-3-large")
embeddings

text="This is a tutorial on OPENAI embedding"
query_result=embeddings.embed_query(text)
query_result

len(query_result)

embeddings_1024=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=1024)

embeddings_1024

text="This is a tutorial on OPENAI embedding"
query_result=embeddings_1024.embed_query(text)
len(query_result)

query_result

from langchain_community.document_loaders import TextLoader

loader=TextLoader('speech.txt')
docs=loader.load()
docs

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
final_documents=text_splitter.split_documents(docs)
final_documents

## Vector Embedding And Vector StoreDB
from langchain_community.vectorstores import Chroma

db=Chroma.from_documents(final_documents,embeddings_1024)
db

### Retrieve the results from query vectorstore db
query="It will be all the easier for us to conduct ourselves as belligerents"
retrieved_results=db.similarity_search(query)
print(retrieved_results)
