from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st

# No need for OpenAI API since we are using Ollama

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please provide response to the user queries."),
        ("user", "Question: {question}")
    ]
)

# Streamlit App
st.title("LangChain + Ollama Demo 💬")

input_text = st.text_input("Ask something:")

# Use local DeepSeek model
llm = Ollama(model="deepseek-r1:1.5b")
output_parser = StrOutputParser()

# Create the LangChain chain
chain = prompt | llm | output_parser

# Handle input
if input_text:
    with st.spinner("Thinking..."):
        response = chain.invoke({'question': input_text})
        st.write(response)
