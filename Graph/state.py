
from typing import TypedDict

class AgentState(TypedDict):
    user_query: str
    route: str
    knowledge_result: str
    coding_result: str
    final_answer: str
