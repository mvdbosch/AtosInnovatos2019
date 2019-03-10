FROM python:3.7

## This handle reaches Marcel van den Bosch
MAINTAINER "Marcel van den Bosch" marcel.vandenbosch@atos.net

RUN mkdir -p /usr/local/AtosInnovatos2019/

WORKDIR /usr/local/AtosInnovatos2019

COPY requirements.txt /usr/local/AtosInnovatos2019/
RUN pip3 install -r requirements.txt
RUN apt-get update -y
RUN apt-get install -y graphviz

COPY data/ /usr/local/AtosInnovatos2019/
ADD data/ /usr/local/AtosInnovatos2019/data/
COPY export/ /usr/local/AtosInnovatos2019/
ADD export/ /usr/local/AtosInnovatos2019/export/
COPY app_images/ /usr/local/AtosInnovatos2019/
ADD app_images/ /usr/local/AtosInnovatos2019/app_images
COPY app.py /usr/local/AtosInnovatos2019/

EXPOSE 8802

CMD ["python","./app.py"]
