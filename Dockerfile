#This is a sample Image 
FROM continuumio/miniconda3
RUN apt update &&\
    apt install -y gcc\
                   g++\
                   make\
                   libxrender1

RUN git clone https://www.github.com/goldmanm/peroxy_branching.git /home/paper_repo

RUN conda env create -f /home/paper_repo/environment.yml
RUN echo "source activate $(head -1 /home/paper_repo/environment.yml | cut -d' ' -f2)" > ~/.bashrc

