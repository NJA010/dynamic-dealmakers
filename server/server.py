from flask import Flask, request, Response
from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os
import time
import random
import logging
from datetime import datetime

load_dotenv()
app = Flask(__name__)
Base = declarative_base()


class Data(Base):
    __tablename__ = 'data'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    value = Column(Float)

@app.route('/simulate', methods=['GET'])
def simulate():
    mode = request.args.get('mode', default = 'db', type = str)

    def generate():
        logging.info(f"Simulation started in {mode} mode")

        if mode.lower() == 'db':
            db_url = os.environ.get("DB_URL")
            
            if not db_url:
                db_user = os.environ.get("TF_VAR_db_user")
                db_pass = os.environ.get("TF_VAR_db_pass")
                db_name = os.environ.get("TF_VAR_db_name")
                cloud_sql_public_ip = os.environ.get("CLOUD_SQL_PUBLIC_IP")

                db_url = f"postgresql://{db_user}:{db_pass}@/{db_name}?host={cloud_sql_public_ip}"

            engine = create_engine(db_url)
            Base.metadata.create_all(engine)
            Session = sessionmaker(bind=engine)
            session = Session()

        for _ in range(10):
            data = Data(value=random.random(), timestamp=datetime.utcnow())
            if mode.lower() == 'db':
                session.add(data)
                session.commit()
                yield f"Data written: {data.value} at {data.timestamp}\n"
            elif mode.lower() == 'log':
                logging.info(f"Data generated: {data.value} at {data.timestamp}")
                yield f"Data generated: {data.value} at {data.timestamp}\n"
            time.sleep(1)
        
        if mode.lower() == 'db':
            session.close()

    return Response(generate(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)