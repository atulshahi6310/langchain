#!/usr/bin/env python
# coding: utf-8

# ## how to split the json data

from langchain_text_splitters import RecursiveJsonSplitter
import json
import requests

url = "https://api.github.com/users/octocat"

response = requests.get(url)
# response.raise_for_status()

data = response.json()

data

json_spliter = RecursiveJsonSplitter(
    max_chunk_size=300,
    
)
json_splits = json_spliter.split_json(data)

json_splits

doc = json_spliter.create_documents(texts=json_splits, metadatas=[{"source": "https://api.github.com/users/octocat"}] * len(json_splits))

doc
