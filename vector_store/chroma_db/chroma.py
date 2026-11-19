#!/usr/bin/env python
# coding: utf-8

# ### ChromaDb
# Chroma is a AI-native open-source vector database focused on developer productivity and happiness. Chroma is licensed under Apache 2.0.

## building a sample vectordb

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loder = TextLoader("k.txt")
data = loder.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data)
docs 

embeddings = OllamaEmbeddings(
model = "nomic-embed-text"     
)
vectorstore = Chroma.from_documents(documents = docs, embedding = embeddings)

query = "why they tall about rivers"
query_result = vectorstore.similarity_search(query)
answer =query_result[0].page_content

print(answer)

## saving the vectorstore to  local machine

vectorDb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="./chroma_db")

## query the db from the local machine 

db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
question = "why they tall about rivers"
query_result = db.similarity_search(question)
answer = query_result[0].page_content
print(answer)

## similarity Search With Score
docs = vectorstore.similarity_search_with_score(query)
docs

### Retriever option
retriever=vectorstore.as_retriever()
retriever.invoke(query)[0].page_content
