
FROM python:3.10.6
EXPOSE 3011
WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD uvicorn fast:app --host 0.0.0.0 --port 3011 --reload
