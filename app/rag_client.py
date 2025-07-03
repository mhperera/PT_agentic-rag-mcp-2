from langchain_mcp_adapters.client import MultiServerMCPClient
from pyfiglet import Figlet
from rich import print

from core.enums.ClassifierLabel import ClassifierLabel
from core.tool_loader import load_tool_config
from core.classifier import question_classifier
from core.agent_factory import build_agents
from core.enums.ToolName import ToolName

f = Figlet(font="banner")


async def rag_cli():

    print(f"[bold magenta]{f.renderText('R A G')}[/bold magenta]")

    tool_config = load_tool_config("config/tool_config.yaml")
    client = MultiServerMCPClient(tool_config)
    
    tools = await client.get_tools()
    tools_map = {getattr(t, "name", f"unnamed_{i}"): t for i, t in enumerate(tools)}
    agents = build_agents(tools_map, tools)

    print("\nType your question (or 'exit'):\n")

    while True:
        question = input("\n🤡 You - ").strip()
        if question.lower() in ["exit", "quit"]:
            print("[green]Exiting...[/green]")
            break

        classifier = await question_classifier(question)

        # print("\nSelected Classifier ::: ", classifier)

        agent = agents.get(
            f"{classifier}_agent", f"{ClassifierLabel.GENERAL_LLM.value}_agent"
        )

        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        last_msg = response["messages"][-1].content
        print("\n🤖 AI -", last_msg)
