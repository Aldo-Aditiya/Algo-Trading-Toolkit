#pytrading:v0.0.2

FROM python:3.7.10

RUN pip install numpy
RUN pip install pandas
RUN pip install matplotlib
RUN pip install sklearn
RUN pip install seaborn
RUN pip install tqdm
RUN pip install quantstats
RUN pip install pandas_ta
RUN pip install statsmodels
RUN pip install alphalens
RUN pip install yfinance
RUN pip install fastapi
RUN pip install uvicorn[standard]
RUN pip install jupyterlab

RUN apt-get update
RUN apt-get install crontab

# Install QOL Items
# RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)"
# RUN chsh -s zsh
RUN apt-get install mc
RUN apt-get install nano