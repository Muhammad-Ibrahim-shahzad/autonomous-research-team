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

