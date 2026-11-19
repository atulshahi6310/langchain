#!/usr/bin/env python
# coding: utf-8

# ## embeddig techinique
# 
# converting the text to vectors

import os 
from dotenv import load_dotenv
load_dotenv() ## load all the enviroment veriables from .env file

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
embeddings

text =  "thi is the war b/w the world best fighters, the battle is going to be very intense and the winner will be the best fighter in the world"
vector = embeddings.embed_query(text)
vector

# ## the above embeding have the variable api key
