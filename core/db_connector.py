from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import inspect
import os
from dotenv import load_dotenv

load_dotenv()

_engine = None
_session = None


def init_db_engine(pool_size=5, max_overflow=10):
    global _engine, _session

    if _engine is None:
        try:
            user = os.getenv("DB_USER", "").strip()
            password = os.getenv("DB_PASSWORD", "").strip()
            host = os.getenv("DB_HOST", "").strip()
            port = os.getenv("DB_PORT", "3306").strip()
            db_name = os.getenv("DB_NAME", "").strip()

            connection_url = (
                f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"
            )

            _engine = create_engine(
                connection_url,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_recycle=1800,
                pool_timeout=30,
                echo=False,
                future=True,
            )

            _session = scoped_session(
                sessionmaker(bind=_engine, autoflush=False, autocommit=False)
            )

        except SQLAlchemyError as e:
            print("- ❌ SQLAlchemy initialization failed:", e)
            raise


def get_session():
    if _engine is None or _session is None:
        init_db_engine()
    return _session()

def get_engine():
    if _engine is None or _session is None:
        init_db_engine()
    return _engine()

def get_db_schema_as_ddl() -> str:
    if _engine is None:
        init_db_engine()

    try:
        inspector = inspect(_engine)
        ddl = ""
        for table in inspector.get_table_names():
            ddl += f"-- Table: {table}\n"
            for column in inspector.get_columns(table):
                ddl += f"  - {column['name']} {column['type']}\n"
        return ddl
    except SQLAlchemyError as e:
        print("- ❌ DB Schema extraction failed:", e)
        raise

def get_db_schema():
    if _engine is None:
        init_db_engine()

    try:
        inspector = inspect(_engine)
        schema = {}

        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            schema[table_name] = []
            for col in columns:
                col_info = {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "default": col.get("default", None),
                    "primary_key": col.get("primary_key", False),
                }
                schema[table_name].append(col_info)
        return schema

    except SQLAlchemyError as e:
        print("- ❌ DB Schema extraction failed:", e)
        raise


def format_schema(schema: dict) -> str:
    lines = []
    for table, cols in schema.items():
        lines.append(f"Table: {table}")
        for col in cols:
            line = f"  - {col['name']} ({col['type']})"
            if col.get("primary_key"):
                line += " [PK]"
            lines.append(line)
    return "\n".join(lines)
