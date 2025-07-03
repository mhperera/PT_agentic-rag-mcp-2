from enum import Enum

class ClassifierLabel(Enum):
    DB_SEARCH = "db_search"
    VECTOR_SEARCH = "vector_search"
    OTHER_TOOL = "other_tool"
    GENERAL_LLM = "general_llm"
    # INTERNET_SEARCH = "internet_search"