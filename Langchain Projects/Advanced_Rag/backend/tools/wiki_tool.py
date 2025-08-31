from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

def get_wiki_tool():
    """Return the Wikipedia search tool configured for concise results."""
    api_wrapper = WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=250
    )
    return WikipediaQueryRun(
        api_wrapper=api_wrapper,
        name="wikipedia",
        description="Useful for searching general knowledge and information about topics on Wikipedia."
    )