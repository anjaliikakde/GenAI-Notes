from langchain_community.tools.arxiv import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

def get_arxiv_tool():
    """Return the Arxiv search tool configured for research papers."""
    arxiv_wrapper = ArxivAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=250
    )
    return ArxivQueryRun(
        api_wrapper=arxiv_wrapper,
        name="arxiv",
        description="Useful for searching and retrieving academic research papers from Arxiv."
    )