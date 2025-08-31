from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import get_settings

settings = get_settings()

def setup_retriever(url):
    """Setup and return a retriever for the given URL."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    
    return vectorstore.as_retriever(search_kwargs={"k": 1})

def get_retriever_tool():
    """Return the dynamic retriever tool for custom documents."""
    return create_retriever_tool(
        retriever=setup_retriever("https://python.langchain.com/docs/get_started/introduction"),
        name="document_retriever",
        description="Useful for retrieving specific information from custom documents or websites. "
                    "Input should be a URL or document reference followed by the query."
    )