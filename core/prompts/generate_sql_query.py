from langchain.prompts import ChatPromptTemplate

def generate_prompt():
    return  ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a SQL and MySQL expert. Given the database schema and a user question, generate a safe SQL SELECT query.\n
                Write ONLY the SQL query (no markdown, no backticks) to answer the following question.\n
                Database tables: {schema}\n
                Question: {question}\n
                SQL Query:"""
            ),
            ("human", "{input}\nSQLQuery:")
        ]
    )
