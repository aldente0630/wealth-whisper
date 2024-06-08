FROM --platform=linux/amd64 python:3.10-slim

COPY app.py requirements.txt ./
COPY utils/ utils/

RUN python3.10 -m pip install --no-cache-dir -r requirements.txt -t .

CMD ["python", "app.py"]