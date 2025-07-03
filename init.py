import os
from core.env_loader import COHERE_API_KEY
from core.db_connector import init_db_engine, get_db_schema
from core.vector_db import read_yaml_file, read_csv_file, read_txt_file
from dotenv import load_dotenv
from yaspin import yaspin
from yaspin.spinners import Spinners
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# llm = ChatGroq(model="qwen-qwq-32b")
# llm = ChatGroq(model="llama3-8b-8192")
llm = ChatGroq(model="llama-3.3-70b-versatile")
# llm = ChatGroq(model="mistral-saba-24b")
# llm = ChatCohere(
#     model_name="xlarge",
#     temperature=0.5,
#     max_tokens=512,
# )

# embedding_model = CohereEmbeddings(
#     cohere_api_key=COHERE_API_KEY, model="embed-english-light-v3.0"
# )
# embedding_model = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     # model="text-embedding-ada-002",
#     chunk_size=1000
# )
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
)


def bootstrap():
    """
    Initialize all resources needed before starting the RAG app.
    - Initialize DB connection
    - Load schemas
    - Load metadata
    if needed
    """

    try:
        with yaspin(Spinners.dots, text="🚀 Bootstrapping resources...") as spinner:

            spinner.text = "🚀 Connecting to SQL database engine..."
            init_db_engine(pool_size=10, max_overflow=10)

            # spinner.text = "🚀 Indexing table metadata from table_info.yaml..."
            # init_vector_db(
            #     file_paths=[
            #         "config/table_info.yaml",
            #     ],
            #     output_file="table_info",
            # )

            # spinner.text = "🚀 Indexing document data (CSV & TXT files)..."
            # init_vector_db(
            #     file_paths=[
            #         # "resources/data.csv",
            #         "resources/data.txt",
            #     ],
            # )

            spinner.text = "🚀 Loading database schema..."
            get_db_schema()

            spinner.text = ""
            spinner.ok("✅ Application Bootstrapping completed.")
    except Exception as e:
        print("❌ Exception:", e)


def init_vector_db(file_paths: list[str], output_file: str = None):
    try:
        documents = []

        for file_path in file_paths:
            ext = os.path.splitext(file_path)[-1].lower()

            if ext in [".yaml", ".yml"]:
                documents.extend(read_yaml_file(file_path))
            elif ext == ".csv":
                documents.extend(read_csv_file(file_path))
            elif ext == ".txt":
                documents.extend(read_txt_file(file_path))
            else:
                print(f"- ⚠️ Unsupported file type: {file_path}")
                continue

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(splits, embedding_model)

        output_path = f"vector_db"

        if output_file:
            output_path += "/" + output_file

        vectorstore.save_local(output_path)

    except Exception as e:
        print("- ❌ Vector DB creation failed:", e)
        raise
