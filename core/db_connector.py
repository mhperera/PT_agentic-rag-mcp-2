from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError
import os
from dotenv import load_dotenv

load_dotenv()

_engine = None
_session = None


def init_db_engine(pool_size=5, max_overflow=10):
    global _engine, _session

    if _engine is None:
        try:
            print("- 🕞 Creating SQLAlchemy DB engine...")
            
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
            print("- ✅ SQLAlchemy engine initialized successfully.")

        except SQLAlchemyError as e:
            print("- ❌ SQLAlchemy initialization failed:", e)
            raise


def get_session():
    if _engine is None or _session is None:
        init_db_engine()
    return _session()
