FROM ubuntu:24.04

RUN apt update && \
    apt -y install --no-install-recommends python3 python3-venv python3-pip python3-setuptools python3-wheel git \
    build-essential libpython3-dev nvidia-cuda-toolkit && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/
WORKDIR /app
COPY scripts/requirements.txt /app
RUN pip install --break-system-packages -r ./requirements.txt

ENTRYPOINT ["/bin/bash"]