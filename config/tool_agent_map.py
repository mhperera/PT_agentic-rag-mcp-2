from core.enums.ClassifierLabel import ClassifierLabel
from core.enums.ToolName import ToolName

def get_tools(tool_map, label: ClassifierLabel):
    return [tool_map[tool.value] for tool in TOOL_AGENT_MAP.get(label, [])]


TOOL_AGENT_MAP: dict[ClassifierLabel, list[ToolName]] = {
    ClassifierLabel.DB_SEARCH: [
        ToolName.VECTOR_TABLE_SEARCH,
        ToolName.GENERATE_SQL_QUERY,
        ToolName.EXECUTE_SQL_QUERY,
        ToolName.REPHRASE_RESULT,
    ],
    ClassifierLabel.OTHER_TOOL: [
        ToolName.MATH_ADD,
        ToolName.MATH_DIVIDE,
        ToolName.MATH_MULTIPLY,
        ToolName.GET_WEATHER,
        ToolName.REPHRASE_RESULT,
    ],
    ClassifierLabel.GENERAL_LLM: [],
    # ClassifierLabel.VECTOR_SEARCH: [
    #     ToolName.VECTOR_KNOWLEDGE_SEARCH,
    # ],
}
