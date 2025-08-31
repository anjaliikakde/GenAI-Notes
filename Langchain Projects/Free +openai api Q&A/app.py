from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please provid response to the user queries"),
        ("user", "Question:{question}")
    ]
)


st.tittle("Langchain demo :")
input_text = st.text_input("Search the topic you want..")

llm = ChatOpenAI(model="gpt-3.5-turbo")

output_parser = StrOutputParser()

chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))