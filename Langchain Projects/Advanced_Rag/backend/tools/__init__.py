from .wiki_tool import get_wiki_tool
from .arxiv_tool import get_arxiv_tool
from .retriever_tool import get_retriever_tool

def get_tools():
    """Get all available tools for the research assistant."""
    return [
        get_wiki_tool(),
        get_arxiv_tool(),
        get_retriever_tool()
    ]