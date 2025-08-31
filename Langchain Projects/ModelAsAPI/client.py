import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

def get_openai_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke",
                             json={'input': {'topic': input_text}})
    
    result = response.json() 

    if 'output' in result:  
        return result['output']['content']
    elif 'detail' in result: 
        return f" Server error: {result['detail']}"
    else:
        return "Unexpected response format"


def get_deepSeek_response(input_text):
    response = requests.post("http://localhost:8000/poem/invoke",
                             json={'input': {'topic': input_text}})
    
    result = response.json() 

    if 'output' in result:  
        return result['output']
    elif 'detail' in result:
        return f"Server error: {result['detail']}"
    else:
        return "Unexpected response format"

st.title('Langchain with OpenAI and Deep Seek API Clients')
input_text = st.text_input("Write an eassy on") 
input_text1 = st.text_input("Write a poem on")  

if input_text:
    st.write(get_openai_response(input_text))
    
if input_text1:
    st.write(get_deepSeek_response(input_text1))
