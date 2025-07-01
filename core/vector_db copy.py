import yaml
import csv
import os
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def init_vector_table_descriptions(
    yaml_path: str, cohere_api_key: str, output_file_name: str = "faiss_index"
):
    try:
        print("- 🕞 Creating FAISS Vector DB...")
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        docs = []
        for table in data:
            content = "\n".join(f"{key}: {value}" for key, value in table.items())
            docs.append(
                Document(
                    page_content=content,
                    metadata={"source": table.get("table_name", "unknown")},
                )
            )

        splits = text_splitter.split_documents(docs)

        embeddings = CohereEmbeddings(
            cohere_api_key=cohere_api_key, model="embed-english-light-v3.0"
        )
        vectorstore = FAISS.from_documents(splits, embeddings)

        output_path = "vector_db/" + output_file_name

        vectorstore.save_local(output_path)
        print(f"- ✅ FAISS vector DB created successfully  at: {output_path}")
    except Exception as e:
        print("- ❌ FAISS Vector DB initialization failed:", e)
        raise

