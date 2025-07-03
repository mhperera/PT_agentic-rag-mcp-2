import re
from core.prompts.classifiers import get_classify_intent_prompt
from core.enums.ClassifierLabel import ClassifierLabel
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from init import llm, embedding_model

examples = [
    # db_search
    {"input": "How many customers are there in the database?", "label": "db_search"},
    {"input": "List all customers from Canada.", "label": "db_search"},
    {"input": "Which customer has the highest credit limit?", "label": "db_search"},
    {"input": "What is the total revenue generated from all orders?","label": "db_search",},
    {"input": "Which product has the highest MSRP?", "label": "db_search"},
    # vector_search
    {"input": "What is solar energy?", "label": "vector_search"},
    {"input": "What are some limitations of using solar power?","label": "vector_search",},
    {"input": "How does solar energy reduce electricity bills?","label": "vector_search",},
    {"input": "What advancements are being made in solar energy technology?","label": "vector_search",},
    {"input": "What is a solar inverter?", "label": "vector_search"},
    {"input": "Why is solar energy considered sustainable?", "label": "vector_search"},
    {"input": "How many states are there in Afghanistan?", "label": "vector_search"},
    {"input": "List all states in Afghanistan.", "label": "vector_search"},
    {"input": "Which states in Afghanistan start with the letter 'B'?","label": "vector_search",},
    {"input": "Do all states have the same country code?", "label": "vector_search"},
    {"input": "Give me all the state codes and names in Afghanistan.","label": "vector_search",},
    {"input": "What is the abbreviation for Kandahar?", "label": "vector_search"},
    # internet_search
    {"input": "What is the capital of America?", "label": "internet_search"},
    {"input": "Who is the President of America?", "label": "internet_search"},
    {"input": "What’s the current price of Tesla stock?", "label": "internet_search"},
    {"input": "What happened in the stock market today?", "label": "internet_search"},
    {"input": "What are the cheapest flights from London to Paris this weekend?","label": "internet_search",},
    {"input": "Are there any openings at Google for front-end engineers?","label": "internet_search",},
    {"input": "What’s trending on Twitter today?", "label": "internet_search"},
    # other_tool
    {"input": "How is the weather in California?", "label": "other_tool"},
    {"input": "What is the current temperature in New York?", "label": "other_tool"},
    {"input": "Will it rain tomorrow in London?", "label": "other_tool"},
    {"input": "What’s the weather forecast for Tokyo this weekend?","label": "other_tool",},
    {"input": "Add 2 and 4", "label": "other_tool"},
    {"input": "Divide 8 from 4", "label": "other_tool"},
    {"input": "Multiply 2 with 2", "label": "other_tool"},
    {"input": "(2 + 4) * 5", "label": "other_tool"},
    # general_llm
    {"input": "Hi, How are you", "label": "general_llm"},
    {"input": "What causes thunderstorms?", "label": "general_llm"},
    {"input": "What’s the difference between “affect” and “effect”?","label": "general_llm",},
    {"input": "Explain what a REST API is.", "label": "general_llm"},
    {"input": "Quiz me on basic Java concepts.", "label": "general_llm"},
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=embedding_model,
    vectorstore_cls=InMemoryVectorStore,
    k=5,
)

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{label}"),
    ]
)

dynamic_few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_selector=example_selector, example_prompt=example_prompt
)


async def question_classifier(question: str) -> str:
    prompt = get_classify_intent_prompt(dynamic_few_shot_prompt).format_messages(
        question=question
    )
    response = await llm.ainvoke(prompt)
    label = response.content.strip().lower()

    # Remove anything between <think>...</think>, including the tags
    label = re.sub(
        r"<think>.*?</think>", "", response.content.strip().lower(), flags=re.DOTALL
    )

    if ClassifierLabel.DB_SEARCH.value in label:
        return ClassifierLabel.DB_SEARCH.value
    elif ClassifierLabel.VECTOR_SEARCH.value in label:
        return ClassifierLabel.VECTOR_SEARCH.value
    # elif ClassifierLabel.INTERNET_SEARCH.value in label:
    #     return ClassifierLabel.INTERNET_SEARCH.value
    elif ClassifierLabel.OTHER_TOOL.value in label:
        return ClassifierLabel.OTHER_TOOL.value
    else:
        return ClassifierLabel.GENERAL_LLM.value
