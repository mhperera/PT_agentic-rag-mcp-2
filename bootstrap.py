from core.env_loader import COHERE_API_KEY
from core.db_connector import init_db_engine, get_db_schema
from core.vector_db import init_vector_db
from dotenv import load_dotenv
from yaspin import yaspin
from yaspin.spinners import Spinners

load_dotenv()


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
            spinner.text = "🚀 Indexing table metadata from table_info.yaml..."
            init_vector_db(
                file_paths=[
                    "config/table_info.yaml",
                ],
                cohere_api_key=COHERE_API_KEY,
                output_file="table_info",
            )
            spinner.text = "🚀 Indexing document data (CSV & TXT files)..."
            init_vector_db(
                file_paths=[
                    "resources/data.csv",
                    "resources/data.txt",
                ],
                cohere_api_key=COHERE_API_KEY,
            )
            spinner.text = "🚀 Loading database schema..."
            get_db_schema()
            spinner.text = ""
            spinner.ok("✅ Application Bootstrapping completed.")
    except Exception as e:
        print("❌ Exception:", e)
