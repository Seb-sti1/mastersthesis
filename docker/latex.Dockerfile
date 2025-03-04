FROM ubuntu:24.04

WORKDIR /app/latex

ENV LANGUAGE=en_US.UTF-8 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

RUN apt update && \
    apt -y install --no-install-recommends pandoc make git procps locales exiftool curl &&\
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen &&\
    locale-gen en_US.UTF-8 &&\
    update-locale LANG=en_US.UTF-8 &&\
    apt -y install texlive texlive-latex-extra biber latexmk \
                   texlive-fonts-extra texlive-science lmodern texlive-fonts-recommended biber &&\
    rm -rf /var/lib/apt/lists/*

RUN curl -L https://cpanmin.us | perl - --self-upgrade &&\
    cpanm Log::Dispatch::File YAML::Tiny File::HomeDir
