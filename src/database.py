import yaml
from sqlalchemy import create_engine, text

queries = yaml.safe_load(open("config/mysql.yaml"))

DB_URL = "postgresql+psycopg2://hema:hemasree123@localhost:5432/bugs_db?options=-csearch_path=public"


def init_db():
    engine = create_engine(DB_URL)

    with engine.begin() as conn:
        # Create main table
        conn.execute(text(queries["create_bugs_table"]))

        # Create embedding tables
        #conn.execute(text(queries["drop_embeddings_tables"]))
        conn.execute(text(queries["create_embeddings_tables"]))

    return engine