from enum import Enum

class ClassifierLabel(Enum):
    GENERAL_LLM = "general_llm"
    DB_SEARCH = "db_search"
    OTHER_TOOL = "other_tool"
    # VECTOR_SEARCH = "vector_search"
    # INTERNET_SEARCH = "internet_search"