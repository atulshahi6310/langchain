#!/usr/bin/env python
# coding: utf-8

# hugging face emmbeding models 

import os 
from dotenv import load_dotenv
load_dotenv() 

os.environ["HUGGINGFACE_ACC"] = os.getenv("HUGGINGFACE_ACC");

from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

text = "this is the text which is written by the author 'atul shahi' he is a very good ai developer and his package is 70 lpa and his per mont salary is around 4.5 lack "

query_result = embedding.embed_query(text);
query_result

len(query_result)

doc_result = embedding.embed_documents([text,"hey do you know that atul shahi is a very good person"])
print(len(doc_result))
len(doc_result[1])

doc_result[1]

para = "this is the paragraph about the india countary which is widely know for the georaphy"
false_emb = embedding.embed_query(para)

false_emb
