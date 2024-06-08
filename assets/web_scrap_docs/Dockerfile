FROM public.ecr.aws/lambda/python:3.10-x86_64

COPY app.py requirements.txt ./
COPY utils/ utils/

RUN python3.10 -m pip install --no-cache-dir -r requirements.txt -t .

CMD ["app.lambda_handler"]