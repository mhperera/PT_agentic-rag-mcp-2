from core.prompts.classifiers import get_classify_intent_prompt
from core.enums.ClassifierLabel import ClassifierLabel
import re


async def question_classifier(llm, question: str) -> str:
    prompt = get_classify_intent_prompt().format_messages(question=question)
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
    elif ClassifierLabel.INTERNET_SEARCH.value in label:
        return ClassifierLabel.INTERNET_SEARCH.value
    elif ClassifierLabel.OTHER_TOOL.value in label:
        return ClassifierLabel.OTHER_TOOL.value
    else:
        return ClassifierLabel.GENERAL_LLM.value
