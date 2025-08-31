# [📑 AxiomAI - Research Assistant]()

This is an advanced RAG-based system that uses tools to search for information. It works like an AI agent that can understand questions, decide which tool to use, gather the information, and provide a helpful answer. It includes three tools: Wikipedia, Arxiv, and a Retriever tool. Together, they allow the system to answer questions by pulling content from Wikipedia, relevant research papers from Arxiv, and any specific documents added by the user. It is helpful for research purposes where reliable and relevant information is needed.

This project helps someone in doing research work by searching and summarizing research papers, general topic explanations, and specific documents. For example, if you input a research topic, it fetches related research papers from Arxiv and provides background information from Wikipedia. The Retriever tool adds the ability to pull in custom documents like LangChain documentation or internal PDFs.

## [AI Agent]()
An AI agent is created by combining a language model with external tools. First, a model like GPT-3.5 is used to understand and respond to natural language. Then, tools are added to give it access to external knowledge or functionalities, such as searching Wikipedia, querying Arxiv, or retrieving documents. Using LangChain, you can build an agent that automatically decides which tool to use for a given user query. The agent takes input, plans its action (which tool to call), performs the action, and finally returns a combined response to the user. This reasoning and acting behavior makes it an AI agent.

## Directory Structure

```
research_assistant/
│
├── .env
├── requirements.txt
├── README.md          # For documentation
│
├── app.py
│
├── config/
│   └── settings.py
│
├── backend/
│   ├── __init__.py
│   ├── llm.py
│   ├── pipeline.py
│   ├── tools/
│      ├── __init__.py
│      ├── wiki_tool.py
│      ├── arxiv_tool.py
│      └── retriever_tool.py
│   
│
├── data/
│   └── faiss_store/               # Saved vector DB
│
├── tests/
│   ├── test_wiki_tool.py
│   ├── test_arxiv_tool.py
│   ├── test_retriever_tool.py
│
└──|

```

## Let's Understand the structure 

**app.py**
This is the main file that runs the Streamlit app. It handles the user interface where people can type queries and see responses. It connects the backend (agent logic) with the frontend.

**backend/**
This folder contains the core logic of your AI agent. It’s separated from the UI so your project is modular and easier to maintain.

**pipeline.py**
This file builds the AI agent by loading the tools (Wikipedia, Arxiv, Retriever) and connecting them with the language model. It’s like the brain of the agent.

**llm.py**
This file configures the OpenAI language model (e.g., GPT-3.5). It sets parameters like temperature and model version. Keeping this separate makes it easier to update or switch models.

- tools/
This folder stores individual tools that the agent can use. Each tool does a specific task.

- retriever_tool.py
Loads a vector store (FAISS) and sets up document retrieval from URLs or uploaded content.

- wiki_tool.py
Fetches information from Wikipedia using LangChain's wrapper.

- arxiv_tool.py
Searches and retrieves research papers from Arxiv.

**data/faiss_store/**
This folder stores FAISS vector indices. These are used to perform fast similarity searches over documents. Saving the index here means it doesn't have to be rebuilt every time.

**config.py**
This file stores configuration values like the OpenAI API key and other settings. Keeping it separate makes the code cleaner and helps manage secrets or environment settings in one place.

**tests/**
Houses test files that verify each tool works correctly and helps catch bugs early.

test_wiki_tool.py, test_arxiv_tool.py, test_retriever_tool.py: Check each tool’s output, handle edge cases, and ensure reliability.

**.env**
Stores environment variables like API keys securely. Keeps sensitive info out of the codebase.

**requirements.txt**
This file lists all Python packages the project needs to run. It helps others install the right dependencies with one command (pip install -r requirements.txt).

This structure keeps the code clean, organized, and easy to scale by separating logic, tools, UI, and config. This makes debugging, testing, and adding new features much simpler.

## Concepts Used in WikiArxiv
## [Vectore Store]()
FAISS (Facebook AI Similarity Search) is a high-performance C++ library with Python bindings, designed for fast nearest-neighbor search in high-dimensional vector spaces, which is critical for semantic search in GenAI applications.

### [WHY?, FAISS]()

**Efficient Indexing & Search:**
FAISS supports multiple indexing strategies like IndexFlatL2, IndexIVFFlat, and HNSW that optimize memory usage and retrieval speed. For example, IVF (Inverted File Index) allows scalable search on millions of embeddings.

**GPU Acceleration:**
FAISS provides optional GPU support, which drastically speeds up vector indexing and query time for large-scale datasets—vital for real-time GenAI applications.

**Local & Offline Capability:**
Unlike cloud-based vector DBs (e.g., Pinecone, Weaviate), FAISS runs fully locally. This reduces latency, ensures privacy, and is ideal for prototypes, air-gapped environments, or when working with sensitive data.

**Customizable and Open-Source:**
FAISS offers full control over index structure, training, and compression (e.g., PQ/OPQ). This is useful in low-resource settings or when building custom LLM pipelines.

**Tight integration with LangChain:**
FAISS is natively supported by LangChain’s retriever interfaces, making it easy to plug into a tool-based agent architecture for document QA or RAG (Retrieval-Augmented Generation) pipelines.

**Storage and Reuse:**
FAISS indices can be persisted to disk (index.save() / load_local() in LangChain), which enables caching, faster reloads, and reduced compute on repeated queries.

### USE OF OTHER VDB

- FAISS → for local, fast, customizable, open-source setups

- Pinecone → better for scalable, distributed, real-time, managed services

- Weaviate → ideal if metadata filtering or hybrid search (text + vector) is needed

- ChromaDB → great for lightweight, LangChain-native workflows.

### [TOOLs]()
Tools are callable function or module that the language model can invoke to perform a specific task it can’t do on its own.

LLMs like GPT can't access the internet or your custom data by default. Tools extend the LLM's capability—letting it search databases, call APIs, or run Python code dynamically.


<img width="1024" height="1536" alt="ChatGPT Image Jul 31, 2025, 10_53_34 AM" src="https://github.com/user-attachments/assets/56f242de-a9b1-46ed-a900-6453b4a3e47b" />

## How to Run it,

### 1. **Clone the Repository**

```bash
git clone https://github.com/your-username/research_assistant.git
cd research_assistant
```

### 2. **Set Up Environment Variables**

Create a `.env` file in the root directory and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. **Create and Activate Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 5. **Run the Streamlit App**

```bash
streamlit run app.py
```

### 6. **Using the App**

* Type a query into the UI.
* The system selects the right tool (Wikipedia / Arxiv / Retriever).
* The LLM responds with a structured answer based on fetched content.

