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

    research_system_prompt = """
    You are a research agent. Your job is to summarize the key facts and 
    findings from the search results below — DO NOT to write the final answer 
    or conclude anything from the search. Just organize what was found: key facts,
    numbers, dates, and claims, clearly and neutrally.
    """

    messages = [
        {
            "role": "system",
            "content": research_system_prompt
        },
        {
            "role": "user",
            "content": f"Search Results: {results}\n\nOriginal Query: {state['query']}"            
        }
    ]
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages
    )
    answer = response.choices[0].message.content

    return {"search_results": results, "research_answer": answer}

def analysis_node(state: MultiAgentState) -> dict:

    analysis_system_prompt = """
    You are a professional analyst. You will be given search results and
    a research summary. Your work is to analyze it properly, organize it 
    in a proper manner, and highlight the most important information.
    DO NOT include anything unnecessary in the final answer. 
    """

    messages = [
        {
            "role": "system",
            "content": analysis_system_prompt
        },
        {
            "role": "user",
            "content": f"Search Results: {state['search_results']}\n\nResearch Summary: {state['research_answer']}"
        }
    ]
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages
    )
    answer = response.choices[0].message.content

    return {"analysis_answer": answer}