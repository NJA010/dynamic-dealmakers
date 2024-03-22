from configparser import ConfigParser
import psycopg2
from psycopg2.extras import Json
import pytz
from datetime import datetime


def load_config(filename='database.ini', section='postgresql') -> dict[str, str]:
    parser = ConfigParser()
    parser.read(filename)

    config = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            config[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return config


def connect(config: dict[str, str]):
    with psycopg2.connect(**config) as conn:
        print('Connected to the PostgreSQL server.')
        return conn


class DatabaseClient:
    def __init__(self, config: dict[str, str]) -> None:
        self.conn = connect(config)
        self.conn.autocommit = True

    def create(self, data: dict, table_name: str):
        with self.conn.cursor() as cur:
            amsterdam_tz = pytz.timezone('Europe/Amsterdam')
            ts = datetime.now(amsterdam_tz)
            columns = ["scraped_at", "payload"]
            columns_str = ', '.join(columns)
            query = f"INSERT INTO {table_name} ({columns_str}) VALUES ('{ts}', {Json(data)});"
            cur.execute(query, data)
            self.conn.commit()
    
    def read(self, query: str) -> dict:
        with self.conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()

    def query_no_return(self, query_string: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(query_string)
            self.conn.commit()

    def read_max_id(self, table_name: str, id_col: str = "id") -> int:
        with self.conn.cursor() as cur:
            query = f"SELECT MAX({id_col}) FROM {table_name};"
            return cur.execute(query).fetchall()[0][0]

    def update(self):
        raise NotImplementedError
    
    def delete(self):
        raise NotImplementedError

    
