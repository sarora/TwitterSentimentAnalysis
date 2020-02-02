FROM python:3.6.7-alpine as intermediate

RUN mkdir -p /home/twitter_sentiment_analysis

WORKDIR "/home/twitter_sentiment_analysis"

COPY . .

RUN pip install -e .[dev]
