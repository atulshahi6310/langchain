#!/usr/bin/env python
# coding: utf-8

# # Getting started With Langchain And Open AI
# 
# In this quickstart we'll see how to:
# 
# - Get setup with LangChain, LangSmith and LangServe
# - Use the most basic and common components of LangChain: prompt templates, models, and output parsers.
# - Build a simple application with LangChain
# - Trace your application with LangSmith
# - Serve your application with LangServe

from langchain_community.document_loaders import PyPDFLoader,TextLoader

loader = TextLoader("linear.txt")
data = loader.load()
print(data[0].page_content)

## reading the pdf file useing the PyPDFLoader

from langchain_community.document_loaders import PyPDFLoader

data1 = PyPDFLoader("a.pdf")
documents = data1.load()
documents

type(documents[0])

for doc in documents:
    with open("web_doc.txt","a") as f:
        f.write(doc.page_content)
        f.write("\n\n")

## web base loaders
from langchain_community.document_loaders import WebBaseLoader
import bs4
data = WebBaseLoader(web_path=("https://en.wikipedia.org/wiki/Linear_regression"),
                     bs_kwargs=dict(parse_only = bs4.SoupStrainer( 
                         class_ = ("mw-body-content","mw-content-ltr mw-parser-output")
                     )))
documents = data.load()

documents

print(len(documents))
type(documents)

print(documents[0].page_content)

import arxiv
from langchain_core.documents import Document

paper_id = "2606.11153"

client = arxiv.Client()

search = arxiv.Search(
    id_list=[paper_id]
)

docs = []

for paper in client.results(search):
    docs.append(
        Document(
            page_content=paper.summary,
            metadata={
                "Title": paper.title,
                "Authors": [author.name for author in paper.authors],
                "Published": paper.published,
                "Entry ID": paper.entry_id,
                "PDF URL": paper.pdf_url,
                "Primary Category": paper.primary_category,
                "Categories": paper.categories,
                
            },
        )
    )

print(len(docs))
print(docs[0].metadata)

print(docs[0].metadata)
print(docs[0].page_content[:10000])
len(docs)
