FROM ros:noetic

RUN apt update && \
    apt -y install --no-install-recommends python3 python3-venv python3-pip python3-setuptools python3-wheel git ros-noetic-rviz && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/
WORKDIR /app
COPY scripts/requirements_ros.txt /app
RUN pip install -r ./requirements_ros.txt