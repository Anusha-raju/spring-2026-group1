import logging
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class DBConnection:
    def __init__(self, host: str, database: str, user: str, password: str, port: int):
        self.conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port,
        )
        self.conn.autocommit = False

    def fetch_all(self, query, params) -> list[dict]:
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                return list(cursor.fetchall())
        except Exception:
            logger.exception("fetch_all failed")
            self.conn.rollback()
            raise

    def fetch_one(self, query: str, params):
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                return cursor.fetchone()
        except Exception:
            logger.exception("fetch_one failed")
            self.conn.rollback()
            raise

    def execute(self, query, params) -> None:
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, params)
            self.conn.commit()
        except Exception:
            logger.exception("execute failed")
            self.conn.rollback()
            raise

    def execute_many(self, query, params_list: list[tuple]) -> None:
        try:
            with self.conn.cursor() as cursor:
                cursor.executemany(query, params_list)
            self.conn.commit()
        except Exception:
            logger.exception("execute_many failed")
            self.conn.rollback()
            raise

    def close(self) -> None:
        if self.conn:
            self.conn.close()
