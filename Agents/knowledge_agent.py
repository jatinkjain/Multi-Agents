from langchain_core.prompts import ChatPromptTemplate
from config import llm
from Graph.state import AgentState

knowledge_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Knowledge Agent specialized in technical theory.

        Your responsibility is to explain concepts, principles,
        architecture, and reasoning — NOT to write code.

        Strict rules:
        1. Explain concepts in simple, clear language.
        2. Focus on what, why, and how things work conceptually.
        3. Use real-world analogies and examples when helpful.
        4. Explain advantages, disadvantages, and use cases when relevant.
        5. Never generate source code, functions, scripts, or
           implementation snippets.
        6. If the user asks for code, do not provide it.
           Explain only the underlying concept or approach.
        7. Do not use code blocks.
        8. Keep the explanation focused on the user's question.
        9. If the query asks for both theory and implementation,
           explain only the theoretical part.

        Your output must contain theory and conceptual explanations only."""
    ),
    ("human", "{user_query}")
])

knowledge_chain = knowledge_prompt | llm 

def knowledge_agent(state: AgentState):
    response = knowledge_chain.invoke({
        "user_query":state["user_query"]
    })

    return {"knowledge_result":response.content}
