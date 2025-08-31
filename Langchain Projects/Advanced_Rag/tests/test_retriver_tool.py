from backend.tools.retriever_tool import setup_retriever

def test_retriever_tool():
    """Test retriever tool returns expected results."""
    retriever = setup_retriever("https://python.langchain.com/docs/get_started/introduction")
    results = retriever.get_relevant_documents("What is LangChain?")
    assert len(results) > 0
    assert "LangChain" in results[0].page_content