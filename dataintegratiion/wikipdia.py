#!/usr/bin/env python
# coding: utf-8

import wikipedia
from langchain_community.document_loaders import WikipediaLoader

# FIX: Set a custom User-Agent before loading
wikipedia.set_user_agent("MyApp/1.0 (`atushahi@gmail.com`)")

data = WikipediaLoader(
    query="cars",
    load_max_docs=5,
    
)
web = data.load()   

for i in web:
    with open("linear.txt","a") as f:
        f.write(i.page_content)
        f.write("\n\n")
