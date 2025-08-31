from backend.tools.wiki_tool import get_wiki_tool

def test_wiki_tool():
    """Test Wikipedia tool returns expected results."""
    tool = get_wiki_tool()
    result = tool.run("LangChain")
    assert isinstance(result, str)
    assert len(result) > 0