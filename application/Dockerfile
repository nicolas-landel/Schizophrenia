# Pull official base image python
FROM python:3.8.0-alpine

# set environment variables
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# set work directory
RUN mkdir code
WORKDIR code

# copy project to /code
COPY . .

RUN apk update \
    && apk add gcc g++ python3-dev musl-dev

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
