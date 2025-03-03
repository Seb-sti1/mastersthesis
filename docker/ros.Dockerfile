FROM ros:noetic

WORKDIR /app
COPY scripts/requirements_ros.txt /app

RUN apt update &&\
    apt -y install --no-install-recommends python3 python3-venv python3-pip python3-setuptools python3-wheel git &&\
    pip install -r ./requirements_ros.txt &&\
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

RUN apt -y install ros-noetic-rviz ros-noetic-rqt-tf-tree ros-noetic-rviz-imu-plugin