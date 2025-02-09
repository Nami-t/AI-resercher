import streamlit as st
import re
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict

# Page Configuration
st.set_page_config(page_title="AI Researcher", page_icon="🧠", layout="wide")

# Inject Custom CSS and JavaScript for Dynamic Input
st.markdown("""
    <style>
        /* Centered Layout */
        .search-container {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        
        /* Search Input Field */
        .search-input {
            transition: width 0.3s ease-in-out;
            min-width: 250px;
            max-width: 600px;
            width: 250px;  /* Default width */
            padding: 12px;
            font-size: 18px;
            border-radius: 10px;
            border: 1px solid #ccc;
        }

        /* Custom Search Button */
        .stButton>button {
            background-color: #007BFF;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 8px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }

        /* Response Box */
        .response-box {
            background: #e8f0fe;
            padding: 15px;
            border-radius: 10px;
            font-size: 16px;
        }
        
    </style>

    <script>
        function adjustWidth() {
            var input = document.getElementById("dynamic-input");
            var length = input.value.length;
            var newWidth = Math.min(250 + length * 7, 600);  // Adjust width dynamically
            input.style.width = newWidth + "px";
        }
    </script>
""", unsafe_allow_html=True)

# Title
st.title("🧠 AI Researcher")

# Search Bar UI
st.markdown('<div class="search-container">', unsafe_allow_html=True)
query = st.text_input("🔎 Enter your research query:", key="search_input", 
                      help="Type your research question. The search bar will expand as you type.")

# Inject JavaScript event listener for dynamic width
st.markdown("""
    <script>
        document.getElementById("dynamic-input").addEventListener("input", adjustWidth);
    </script>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Search Button with Centered Layout
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    search_clicked = st.button("🔍 Search")

# Templates for Summarization & Response Generation
summary_template = """
Summarize the following content into a concise paragraph that directly addresses the query. Ensure the summary 
highlights the key points relevant to the query while maintaining clarity and completeness.
Query: {query}
Content: {content}
"""

generate_response_template = """    
Given the following user query and content, generate a response that directly answers the query using relevant 
information from the content. Ensure that the response is clear, concise, and well-structured. 
Additionally, provide a brief summary of the key points from the response. 
Question: {question} 
Context: {context} 
Answer:
"""

# Define Data Structures
class ResearchState(TypedDict):
    query: str
    sources: list[str]
    web_results: list[str]
    summarized_results: list[str]
    response: str

class ResearchStateInput(TypedDict):
    query: str

class ResearchStateOutput(TypedDict):
    sources: list[str]
    response: str

# Search Web Function
def search_web(state: ResearchState):
    search = TavilySearchResults(max_results=3, tavily_api_key="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")
    search_results = search.invoke(state["query"])

    return  {
        "sources": [result['url'] for result in search_results],
        "web_results": [result['content'] for result in search_results]
    }

# Summarization Function
def summarize_results(state: ResearchState):
    model = ChatOllama(model="deepseek-r1:8b")
    prompt = ChatPromptTemplate.from_template(summary_template)
    chain = prompt | model

    summarized_results = []
    for content in state["web_results"]:
        summary = chain.invoke({"query": state["query"], "content": content})
        clean_content = clean_text(summary.content)
        summarized_results.append(clean_content)

    return {
        "summarized_results": summarized_results
    }

# Response Generation Function
def generate_response(state: ResearchState):
    model = ChatOllama(model="deepseek-r1:8b")
    prompt = ChatPromptTemplate.from_template(generate_response_template)
    chain = prompt | model

    content = "\n\n".join([summary for summary in state["summarized_results"]])

    return {
        "response": chain.invoke({"question": state["query"], "context": content})
    }

# Text Cleaning Function
def clean_text(text: str):
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

# Build Graph Pipeline
builder = StateGraph(
    ResearchState,
    input=ResearchStateInput,
    output=ResearchStateOutput
)

builder.add_node("search_web", search_web)
builder.add_node("summarize_results", summarize_results)
builder.add_node("generate_response", generate_response)

builder.add_edge(START, "search_web")
builder.add_edge("search_web", "summarize_results")
builder.add_edge("summarize_results", "generate_response")
builder.add_edge("generate_response", END)

graph = builder.compile()

# Run the search if button is clicked
if search_clicked:
    if query:
        with st.spinner("🔍 Searching and analyzing..."):
            response_state = graph.invoke({"query": query})

        st.subheader("📝 AI-Generated Response:")
        st.markdown(f"<div class='response-box'>{clean_text(response_state['response'].content)}</div>", unsafe_allow_html=True)

        st.subheader("🔗 Sources:")
        for source in response_state["sources"]:
            st.markdown(f"✅ [Source]({source})", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a query first!")
