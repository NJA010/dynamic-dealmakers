from configparser import ConfigParser
import psycopg2


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

    def create(self):
        raise NotImplementedError
    
    def read(self):
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError
    
    def delete(self):
        raise NotImplementedError
    