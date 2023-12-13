FROM python:3.11-slim-buster

WORKDIR /app
COPY ./requirements.txt /app/
RUN pip3 install -r requirements.txt
COPY . .
EXPOSE 5000
ENV FLASK_APP=server.py

CMD [ "flask", "run", "--host", "0.0.0.0" ]