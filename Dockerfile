FROM python:3.11-slim

RUN apt update && \
    apt install pipx -y && \
    pipx ensurepat#h

WORKDIR /app

ADD . /app

WORKDIR /app

RUN make install

CMD ["make", "start"]
