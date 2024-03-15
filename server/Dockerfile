FROM python:3.11.4-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache -r requirements.txt

COPY . .

RUN pip install -e .

EXPOSE 8080

CMD [ "python", "server.py" ]