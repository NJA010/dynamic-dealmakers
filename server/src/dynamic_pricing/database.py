from configparser import ConfigParser
from typing import Any
import os 

import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, execute_values
import pytz
from datetime import datetime
from pathlib import Path


from dynamic_pricing.env_setting import define_app_creds, get_secret

DEBUG = os.getenv('DEBUG', True) in ['true', 'True', True]
PROJECT_ID = os.getenv('PROJECT_ID')
SECRET_ID = os.getenv('SECRET_ID')
VERSION_ID = os.getenv('VERSION_ID')

def load_config(filename='database.ini', section='postgresql') -> dict[str, str]:
    _ = define_app_creds()
    if not DEBUG:
        # run in cloud
        return get_secret(
            project_id=PROJECT_ID,
            secret_id=SECRET_ID,
            version_id=VERSION_ID,
            )        

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
            cur.execute(query)
            return cur.fetchall()[0][0]

    def insert_values(self, table_name: str, values: list[list[Any]], column_names: list[str]) -> None:
        with self.conn.cursor() as cur:
            insert_query = sql.SQL(f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES %s")
            execute_values(cur, insert_query, values)
            self.conn.commit()

    def read_df(self, query: str) -> (dict, list):
        with self.conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall(), cur.description

    def update(self):
        raise NotImplementedError
    
    def delete(self):
        raise NotImplementedError
