import os
import streamlit as st
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

## load the enviroment variables 
load_dotenv();

#  this is for the langsmith tracking 
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
#  lanchain trackeing 
os.environ["LANGCHAIN_TRACING_V2"] = "true";

os.environ["LANCHAIN_PROJECT_NAME"] = os.getenv("LANCHAIN_PROJECT_NAME")
print("loadding complete")

## promt tamplete 
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate([
    (
        "system",
       "tell what every he want about the project and give the answer in a simple way"
    ),
    (
        "user",
        "{question}"
    )
])

## streamlit framework
st.title("I am a rag app how i can i help you")
input_text=st.text_input("What question you have in mind?")

## ollama model calling 

llm = ChatOllama(
    model="gemma3:1b",
    temperature=0
)

output = StrOutputParser()
chain = prompt|llm|output

if input_text:
    st.write(chain.invoke({"question":input_text}))
    st.write("thanks for the visiting")
    
