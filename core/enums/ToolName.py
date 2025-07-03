from enum import Enum

class ToolName(Enum):
    MATH_ADD = "math_add"
    MATH_MULTIPLY = "math_multiply"
    MATH_DIVIDE = "math_divide"
    GET_WEATHER = "get_weather"

    GENERATE_SQL_QUERY = "generate_sql_query"
    EXECUTE_SQL_QUERY = "execute_sql_query"
    REPHRASE_SQL_RESULT = "rephrase_sql_result"

    VECTOR_KNOWLEDGE_SEARCH = "vector_knowledge_search"
    VECTOR_TABLE_SEARCH = "vector_table_search"

    REPHRASE_RESULT = "rephrase_result"
