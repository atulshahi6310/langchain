#!/usr/bin/env python
# coding: utf-8

# ### how to split by char  - chr text spliter

from langchain_community.document_loaders import TextLoader

loader = TextLoader("story.txt")
documents = loader.load()
print(documents)

from langchain_text_splitters import CharacterTextSplitter

text_spliter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size =100,chunk_overlap =20

)
text_spliter.split_documents(documents)
