from core.prompts.classifiers import get_classify_intent_prompt
from core.enums.IntentLabel import IntentLabel
import re


async def question_classifier(llm, question: str) -> str:
    prompt = get_classify_intent_prompt().format_messages(question=question)
    response = await llm.ainvoke(prompt)
    label = response.content.strip().lower()

    # Remove anything between <think>...</think>, including the tags
    label = re.sub(
        r"<think>.*?</think>", "", response.content.strip().lower(), flags=re.DOTALL
    )

    if IntentLabel.DB_SEARCH.value in label:
        return IntentLabel.DB_SEARCH.value
    elif IntentLabel.VECTOR_SEARCH.value in label:
        return IntentLabel.VECTOR_SEARCH.value
    elif IntentLabel.INTERNET_SEARCH.value in label:
        return IntentLabel.INTERNET_SEARCH.value
    elif IntentLabel.OTHER_TOOL.value in label:
        return IntentLabel.OTHER_TOOL.value
    else:
        return IntentLabel.GENERAL_LLM.value
