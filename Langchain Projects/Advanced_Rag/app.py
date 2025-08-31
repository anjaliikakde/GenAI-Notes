"""
Research Assistant Streamlit Application

A comprehensive AI-powered research assistant with tools for Wikipedia, Arxiv, 
and custom document retrieval. Features a modern chat interface with conversation
history, loading states, and responsive design.
"""

import streamlit as st
from dotenv import load_dotenv
from backend.pipeline import setup_research_assistant

# Configure page settings
st.set_page_config(
    page_title="Research Assistant",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_assistant():
    """
    Initialize and cache the research assistant.
    
    Returns:
        AgentExecutor: Configured research assistant agent
    """
    load_dotenv()
    return setup_research_assistant()

def display_chat_message(role, content):
    """
    Display a chat message with appropriate styling.
    
    Args:
        role (str): 'user' or 'assistant'
        content (str): Message content
    """
    with st.chat_message(role):
        st.markdown(content)

def main():
    """
    Main Streamlit application function.
    Handles UI rendering and chat functionality.
    """
    # Custom CSS for styling
    st.markdown("""
    <style>
        .stChatInput textarea {
            min-height: 120px !important;
        }
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .user-message {
            background-color: #f0f2f6;
        }
        .assistant-message {
            background-color: #e6f7ff;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.title("Settings")
        st.markdown("Configure your research tools")
        
        tools_enabled = st.multiselect(
            "Active Research Tools",
            options=["Wikipedia", "Arxiv", "Document Retriever"],
            default=["Wikipedia", "Arxiv", "Document Retriever"],
            help="Select which research tools to enable"
        )
        
        st.markdown("---")
        st.markdown("**About**")
        st.markdown("""
        This AI research assistant can:
        - Search Wikipedia for general knowledge
        - Find relevant Arxiv research papers
        - Retrieve information from your custom documents
        """)

    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("📑 AxiomAI - AI Research Assistant")
        st.caption("Powered by Wikipedia, Arxiv, and custom document retrieval")
        
        # Initialize session state
        if "assistant" not in st.session_state:
            with st.spinner("Initializing research tools..."):
                st.session_state.assistant = initialize_assistant()
                st.session_state.tools_enabled = tools_enabled

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm your research assistant. What would you like to learn about today?"}
            ]

        # Display chat history
        for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])

        # Chat input
        if prompt := st.chat_input("Enter your research question..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            display_chat_message("user", prompt)

            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Researching..."):
                    try:
                        response = st.session_state.assistant.invoke({"input": prompt})
                        response_content = response['output']
                    except Exception as e:
                        response_content = f"Sorry, I encountered an error: {str(e)}"
                
                # Stream the response
                message_placeholder = st.empty()
                full_response = ""
                for chunk in response_content.split():
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_content})

    with col2:
        st.markdown("### 📚 Research Sources")
        if st.session_state.get("messages", []):
            last_query = st.session_state.messages[-2]["content"] if len(st.session_state.messages) > 1 else ""
            last_response = st.session_state.messages[-1]["content"] if st.session_state.messages else ""
            
            with st.expander("Last Response Analysis"):
                st.markdown(f"**Query:** {last_query}")
                st.markdown("**Sources Used:**")
                
                if "Wikipedia" in tools_enabled and "wiki" in last_response.lower():
                    st.success("Wikipedia")
                if "Arxiv" in tools_enabled and "arxiv" in last_response.lower():
                    st.info("Arxiv")
                if "Document Retriever" in tools_enabled and "document" in last_response.lower():
                    st.warning("Custom Documents")
        
        st.markdown("---")
        st.markdown("**💡 Tips**")
        st.markdown("""
        - Be specific with your questions
        - Try asking about recent research papers
        - Upload documents for custom retrieval
        """)

if __name__ == "__main__":
    main()