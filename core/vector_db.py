import yaml
import csv
import os
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def init_vector_db(file_paths: list[str], cohere_api_key: str):
    try:
        print("- 🕞 Creating a single combined FAISS Vector DB from multiple files...")

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

        embeddings = CohereEmbeddings(
            cohere_api_key=cohere_api_key, model="embed-english-light-v3.0"
        )
        vectorstore = FAISS.from_documents(splits, embeddings)

        output_path = f"vector_db"
        vectorstore.save_local(output_path)

        print(f"- ✅ Single combined FAISS vector DB created at: {output_path}")

    except Exception as e:
        print("- ❌ Vector DB creation failed:", e)
        raise


def read_yaml_file(file_path: str) -> list[Document]:
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    documents = []
    for item in data:
        content = flatten_dict(item)
        documents.append(
            Document(
                page_content=content, metadata={"source": os.path.basename(file_path)}
            )
        )
    return documents


def read_csv_file(file_path: str) -> list[Document]:
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = flatten_dict(row)
            documents.append(
                Document(
                    page_content=content,
                    metadata={"source": os.path.basename(file_path)},
                )
            )
    return documents


def read_txt_file(file_path: str) -> list[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return [
        Document(page_content=content, metadata={"source": os.path.basename(file_path)})
    ]


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> str:
    """
    Recursively flattens a nested dictionary and returns as plain text.
    """
    items = []

    def _flatten(obj, parent):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(v, f"{parent}{sep}{k}" if parent else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _flatten(v, f"{parent}[{i}]")
        else:
            items.append(f"{parent}: {obj}")

    _flatten(d, parent_key)
    return "\n".join(items)
