from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from dotenv import load_dotenv
from typing import TypedDict
from groq import Groq
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

class MultiAgentState(TypedDict):
    query: str
    search_results: list[dict]
    research_answer: str
    analysis_answer: str
    report_answer: str
    total_cost: float

def research_node(state: MultiAgentState) -> dict:

    query = state["query"]
    search_results = tavily_client.search(query=query)
    results = search_results["results"]

    draft_system_prompt = """
    You are a research assistant, answer the question using the search results below.
    """
    messages = [
        {
            "role": "system",
            "content": draft_system_prompt
        }
        {
            "role": "user",
            "content": f"Search results: "
        }
    ]