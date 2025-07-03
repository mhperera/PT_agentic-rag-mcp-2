from langchain.prompts import ChatPromptTemplate


def generate_prompt():
    return ChatPromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question.\n
        Question: {question}\n
        SQL Query: {sql_query}\n
        SQL Result: {result}\n
        Answer: """
    )
