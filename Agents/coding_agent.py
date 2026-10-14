from langchain_core.prompts import ChatPromptTemplate
from config import llm
from Graph.state import AgentState

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

coding_chain = coding_prompt | llm 

def coding_agent(state:AgentState):
    response = coding_chain.invoke({
        "user_query":state["user_query"]
    })

    return {"coding_result":response.content}
