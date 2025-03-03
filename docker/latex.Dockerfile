FROM ubuntu:24.04

RUN apt update && \
    apt -y install --no-install-recommends pandoc texlive texlive-latex-extra biber latexmk make git procps locales curl && \
    rm -rf /var/lib/apt/lists/*

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

RUN curl -L https://cpanmin.us | perl - --self-upgrade && \
    cpanm Log::Dispatch::File YAML::Tiny File::HomeDir

RUN apt update && \
    apt -y install --no-install-recommends texlive-fonts-extra texlive-science lmodern texlive-fonts-recommended && \
    rm -rf /var/lib/apt/lists/*

RUN apt update &&\
    apt -y install exiftool && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/latex

WORKDIR /app/latex