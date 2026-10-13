from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()


class AgentState(TypedDict):
    user_query: str
    route: str
    knowledge_result: str
    coding_result: str
    final_answer: str

class RouteDecision(BaseModel):
    route: str = Field(
        description="Choose exactly one: knowledge, coding, both"
    )

llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)

router_llm = llm.with_structured_output(RouteDecision)

router_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Classify the user's query.

        Choose:
        - knowledge: concepts, explanations, research
        - coding: code generation, debugging, implementation
        - both: the query needs both explanation and coding

        Return only the structured route decision."""
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


knowledge_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Knowledge Agent.
        Explain technical concepts clearly and accurately.
        Use examples when helpful.
        Focus on concepts and reasoning, not full code implementation."""
    ),
    ("human", "{user_query}")
])

coding_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Coding Agent.

        Your primary responsibility is to solve programming tasks
        by writing code, not by giving lengthy theoretical explanations.

        Rules:
        1. For code-generation tasks, provide the complete working code.
        2. For debugging tasks, identify the issue briefly and provide
           the corrected code.
        3. For implementation tasks, focus on the actual implementation.
        4. Keep explanations to a maximum of 2-3 short sentences.
        5. Do not explain general concepts unless explicitly asked.
        6. Use Markdown code blocks with the correct language.
        7. If the user asks for code only, return code only.
        8. If requirements are unclear, ask one concise question.
        """
    ),
    ("human", "{user_query}")
])

knowledge_chain = knowledge_prompt | llm 
coding_chain = coding_prompt | llm 

def knowledge_agent(state: AgentState):
    response = knowledge_chain.invoke({
        "user_query":state["user_query"]
    })

    return {"knowledge_result":response.content}

def coding_agent(state:AgentState):
    response = coding_chain.invoke({
        "user_query":state["user_query"]
    })

    return {"coding_result":response.content}

def aggregator_node(state: AgentState):
    knowledge = state["knowledge_result"].strip()
    coding = state["coding_result"].strip()

    if knowledge and coding:
        final_answer = (
            f"## Knowledge\n{knowledge}\n\n"
            f"## Coding Solution\n{coding}"
        )
    elif knowledge:
        final_answer = knowledge
    elif coding:
        final_answer = coding
    else:
        final_answer = "No result was generated."

    return {"final_answer": final_answer}

def parallel_node(state: AgentState):
    return {}



def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("router", router_node)
    builder.add_node("knowledge_only", knowledge_agent)
    builder.add_node("knowledge_parallel", knowledge_agent)
    builder.add_node("coding_only",coding_agent)
    builder.add_node("coding_parallel",coding_agent)
    builder.add_node("parallel", parallel_node)
    builder.add_node("aggregator", aggregator_node)
    builder.add_edge(START, "router")
    builder.add_conditional_edges("router",
                                route_decision,{
        "knowledge_only": "knowledge_only",
        "coding_only": "coding_only",
        "parallel": "parallel"
    })
    builder.add_edge("knowledge_only", "aggregator")
    builder.add_edge("coding_only", "aggregator")
    builder.add_edge("parallel", "knowledge_parallel")
    builder.add_edge("parallel", "coding_parallel")
    builder.add_edge(
        ["knowledge_parallel", "coding_parallel"],
        "aggregator"
    )

    builder.add_edge("aggregator", END)

    return builder.compile()



def main():
    graph = build_graph()

    user_query = input("You: ").strip()

    result = graph.invoke({"user_query": user_query})

    print("\nAssistant:")
    print(result["final_answer"])
    
    

if __name__=="__main__":
    main()