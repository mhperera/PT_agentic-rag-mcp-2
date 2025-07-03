import yaml
import csv
import os
from langchain_core.documents import Document


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
