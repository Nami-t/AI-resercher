import streamlit as st
import re
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict

# Styling for better UI
st.set_page_config(page_title="AI Researcher", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f4;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            padding: 10px;
            font-size: 18px;
        }
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
        .stMarkdown {
            font-size: 16px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

summary_template = """
Summarize the following content into a concise paragraph that directly addresses the query. 
Ensure the summary highlights the key points relevant to the query while maintaining clarity and completeness.
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

# Function to search the web using Tavily
def search_web(state: ResearchState):
    search = TavilySearchResults(max_results=3)  # API key should be in ENV
    search_results = search.invoke(state["query"])
    
    return {
        "sources": [result['url'] for result in search_results],
        "web_results": [result['content'] for result in search_results]
    }

# Function to summarize results
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

# Function to generate response
def generate_response(state: ResearchState):
    model = ChatOllama(model="deepseek-r1:8b")
    prompt = ChatPromptTemplate.from_template(generate_response_template)
    chain = prompt | model

    content = "\n\n".join(state["summarized_results"])

    return {
        "response": chain.invoke({"question": state["query"], "context": content})
    }

# Function to clean the text output
def clean_text(text: str):
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

# Define the AI Researcher Graph
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

# Streamlit UI
st.title("🧠 AI Researcher")

query = st.text_input("🔎 Enter your research query:")

if st.button("Search"):
    if query:
        with st.spinner("Searching and analyzing..."):
            response_state = graph.invoke({"query": query})

        st.subheader("📝 AI-Generated Response:")
        st.markdown(f"<div style='padding:10px; border-radius:10px; background:#e8f0fe; font-size:16px;'>{clean_text(response_state['response'].content)}</div>", unsafe_allow_html=True)

        st.subheader("🔗 Sources:")
        for source in response_state["sources"]:
            st.markdown(f"✅ [Source]({source})", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a query first!")
