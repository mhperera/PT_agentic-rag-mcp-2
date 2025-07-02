from enum import Enum

class IntentLabel(Enum):
    DB_SEARCH = "db_search"
    VECTOR_SEARCH = "vector_search"
    INTERNET_SEARCH = "internet_search"
    OTHER_TOOL = "other_tool"
    GENERAL_LLM = "general_llm"