FROM python:3.11.4-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache -r requirements.txt

COPY . .

EXPOSE 8080

CMD [ "python", "simulate.py" ]