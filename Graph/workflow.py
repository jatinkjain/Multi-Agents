from langgraph.graph import StateGraph,START,END
from Graph.state import AgentState
from Agents.knowledge_agent import knowledge_agent
from Agents.coding_agent import coding_agent
from router import route_decision, router_node


def aggregator_node(state: AgentState):  
    knowledge = state.get("knowledge_result", "").strip()
    coding = state.get("coding_result", "").strip()

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
