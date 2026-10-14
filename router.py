from pydantic import BaseModel
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from Graph.state import AgentState

class RouteDecision(BaseModel):
    route: Literal["knowledge","coding","both"]


router_llm = llm.with_structured_output(RouteDecision)


router_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Query Routing Classifier for a Multi-Agent Technical Assistant.

Your ONLY responsibility is to classify the user's query into exactly one route:
- knowledge
- coding
- both

STRICT ROUTING RULES:

1. KNOWLEDGE:
   Choose "knowledge" when the user wants:
   - Conceptual explanations or theory
   - Definitions, comparisons, advantages, or disadvantages
   - Architecture, workflows, or how something works
   - Technical research or conceptual reasoning

   The Knowledge Agent must provide theory only.
   It must NOT generate code.

2. CODING:
   Choose "coding" when the user wants:
   - Source code or a complete implementation
   - Code debugging, fixing, or refactoring
   - Programming functions, scripts, or algorithms
   - Changes to existing code

   The Coding Agent must focus on implementation.

3. BOTH:
   Choose "both" ONLY when the user explicitly needs
   BOTH a conceptual explanation AND code/implementation.

   Examples:
   - "Explain RAG and implement a basic RAG pipeline."
   - "Explain recursion and write a recursive Python function."

IMPORTANT RULES:
- Do NOT answer the user's query.
- Do NOT generate code.
- Do NOT call or select agents yourself.
- Return only the structured route decision.
- Do NOT choose "both" merely because a coding task
  could benefit from an explanation.
- If the request is purely theoretical, choose "knowledge".
- If the request is purely implementation-focused, choose "coding".
- Classify based on what the user asks for, not what you
  think they might need."""
    ),
    ("human", "{user_query}")
])

def router_node(state: AgentState):
    decision = (router_prompt | router_llm).invoke({
        "user_query": state["user_query"]
    })

    return {"route": decision.route}


def route_decision(state: AgentState):
    route= state["route"]

    if route == "knowledge":
        return "knowledge_only"

    elif route == "coding":
        return "coding_only"

    elif route == "both":
        return "parallel"

    else :
        raise ValueError(f"Invalid route: {route}") 
