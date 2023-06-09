FROM --platform=x86_64 python:3.9
COPY . /app
WORKDIR /app
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
RUN bash miniconda.sh -b -p /usr/local/Miniconda
RUN eval "$(/usr/local/Miniconda/bin/conda shell.bash hook)" \
    && conda env create --name primekg --file environment.yml

CMD eval "$(/usr/local/Miniconda/bin/conda shell.bash hook)" \
    && conda activate primekg \
    && jupyter notebook --allow-root --ip 0.0.0.0