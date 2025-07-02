def get_llm():
    model = os.getenv("MODEL_NAME", "qwen-qwq-32b")
    return ChatGroq(model=model)
