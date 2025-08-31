from backend.tools.arxiv_tool import get_arxiv_tool

def test_arxiv_tool():
    """Test Arxiv tool returns expected results."""
    tool = get_arxiv_tool()
    result = tool.run("large language models")
    assert isinstance(result, str)
    assert len(result) > 0