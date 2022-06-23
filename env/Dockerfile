#pytrading:v0.0.4

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
RUN pip install arch
RUN pip install alphalens
RUN pip install yfinance
RUN pip install fastapi
RUN pip install uvicorn[standard]
RUN pip install jupyterlab

# Install R
RUN apt install dirmngr gnupg apt-transport-https ca-certificates software-properties-common -y
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'
RUN apt install r-base -y
RUN pip install rpy2[all]

RUN apt-get update
RUN apt-get install cron -y

# Install QOL Items
# RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)"
# RUN chsh -s zsh
RUN apt-get install mc -y
RUN apt-get install nano -y