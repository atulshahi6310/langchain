#!/usr/bin/env python
# coding: utf-8

# ## how to reqursevily split the text by  char

from langchain_community.document_loaders import TextLoader

loader = TextLoader("/run/media/atul/New Volume/gen-ai/langchain/dataintegratiion/linear.txt")
docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=20
    
    )

texts = text_splitter.create_documents([doc.page_content for doc in docs])

print(texts[0].page_content)
print(texts[1].page_content)
print(texts[2].page_content)

# ## second expample 

from langchain_community.document_loaders import TextLoader

texts = TextLoader("story.txt").load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
texts = text_splitter.create_documents([text.page_content for text in texts])

for i in range(len(texts)):
    print(texts[i].page_content) 
    print("\n\n")
