#!/usr/bin/env python
# coding: utf-8

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

#   this donot work beacuse we don't have a api key 
# # take the input from the user ??
# result = llm.invoke("who is gpt??");
# print(result)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert AI developer. Provide accurate, clear, and concise answers based on the user's questions."
        ),
        (
            "user",
            "{input}"
        )
    ]
)

print(prompt)

#chain for llm know what is have to behave like
# chain is meanns we can combine multiple thing like promt then llm and what ever we can add

chain = prompt|llm

# responce = chain.invoke({"input":"can you tell me about the langsmith  ?? "})
# print(responce) 

## string output paerser  --> getting the massage from the llm and how we are want to display it 

from langchain_core.output_parsers import StrOutputParser
output_parsers = StrOutputParser();
chain = prompt|llm|output_parsers #// this is the chain
