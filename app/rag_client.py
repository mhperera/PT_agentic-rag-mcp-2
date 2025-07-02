from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere

from pyfiglet import Figlet
from rich import print
from core.tool_loader import load_tool_config
from core.classifier import question_classifier
from core.agent_factory import build_agents
from core.enums.ClassifierLabel import ClassifierLabel

f = Figlet(font="banner")

llm = ChatGroq(model="qwen-qwq-32b")
# llm = ChatGroq(model="llama3-8b-8192")
# llm = ChatGroq(model="llama-3.3-70b-versatile")
# llm = ChatGroq(model="mistral-saba-24b")
# llm = ChatCohere(
#     model_name="xlarge",
#     temperature=0.5,
#     max_tokens=512,
# )


async def rag_cli():

    print(f"[bold magenta]{f.renderText('R A G')}[/bold magenta]")

    tool_config = load_tool_config("config/tool_config.yaml")
    client = MultiServerMCPClient(tool_config)
    tools = await client.get_tools()
    tool_map = {getattr(t, "name", f"unnamed_{i}"): t for i, t in enumerate(tools)}
    llm_with_tools = llm.bind_tools(tools)
    agents = build_agents(llm_with_tools, tools, tool_map)

    print("\nType your question (or 'exit'):\n")

    while True:
        question = input("\n🤡 You - ").strip()
        if question.lower() in ["exit", "quit"]:
            print("[green]Exiting...[/green]")
            break

        classifier = await question_classifier(llm, question)

        print("\nSelected Classifier ::: ", classifier)

        agent = agents.get(
            f"{classifier}_agent", f"{ClassifierLabel.OTHER_TOOL.value}_agent"
        )

        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        last_msg = response["messages"][-1].content
        print("\n🤖 AI -", last_msg)
