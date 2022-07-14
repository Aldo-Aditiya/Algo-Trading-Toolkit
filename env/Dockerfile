#pytrading:v0.0.6

FROM brunneis/python:3.8.3-ubuntu-20.04
ENTRYPOINT /bin/bash

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
RUN pip install yfinance
RUN pip install python-binance
RUN pip install fastapi
RUN pip install uvicorn[standard]
RUN pip install jupyterlab
RUN pip install swifter
RUN pip install papermill
RUN pip install mlflow
RUN pip install black

# Install R
RUN apt-get update
RUN apt install dirmngr gnupg apt-transport-https ca-certificates software-properties-common -y
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'
RUN apt-get update
RUN apt-get install r-base -y
RUN pip install rpy2[all]
RUN apt-get install curl libcurl4-openssl-dev libssl-dev -y

# Install R Packages
RUN R -e "install.packages('quantmod')"
RUN R -e "install.packages('timeSeries')"
RUN R -e "install.packages('rugarch')"
RUN R -e "install.packages('lattice')"
RUN R -e "install.packages('IRkernel')"
RUN R -e "IRkernel::installspec(user = FALSE)"

RUN apt-get update
RUN apt-get install cron -y
RUN apt-get install git -y

# Install QOL Items
# RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)"
# RUN chsh -s zsh
RUN apt-get install mc -y
RUN apt-get install nano -y


