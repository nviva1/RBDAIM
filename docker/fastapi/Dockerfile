FROM python:3.10.0

RUN mkdir /src
RUN mkdir /src/logs

COPY ./fastapi/requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

COPY ./fastapi/server.py ./fastapi/.env src/

WORKDIR /src

CMD python server.py