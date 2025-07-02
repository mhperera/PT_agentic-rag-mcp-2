from langchain_core.prompts import ChatPromptTemplate

def get_classify_intent_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an Expert Classification Assistant.\n"
                "Given a user question, classify it into exactly one of the following categories:\n\n"
                "1. db_search - Question requires SQL data from a database. Database is about sales, orders, customers, payments etc\n"
                "2. vector_search - Question requires information from indexed documents. Specially about solar energy and Educational qualifications\n"
                "3. internet_search - Question requires current or external web knowledge.\n"
                "4. other_tool - Question is best handled by a non-LLM tool (like calculator, plotter).\n"
                "5. general_llm - General reasoning or chit-chat.\n\n"
                "**Respond with only the category label**, such as 'db_search' or 'other_tool' etc, and nothing else.\n"
                "Do not explain. Do not use punctuation. Return only one of the labels above. Nothing else."
            ),
            ("human", "{question}"),
        ]
    )
