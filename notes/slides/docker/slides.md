---
theme: default
paginate: true
title: How to use docker?
description: How to use docker?
author: Sébastien Kerbourc'h
style: |
  section {
    font-size: 20px;
  }
  .center {
    text-align: center;
  }
  .sidenote {
    color: gray;
    font-style: italic;
  }
header: 'How to use docker?'
footer: '<img height="50px" src="../../../latex/logo/enstaparis_old.jpg"/>&nbsp;<img height="50px" src="../../../latex/logo/dtu.png"/>
  &nbsp;&nbsp;
  _Slides available on my [GitHub](https://github.com/Seb-sti1/mastersthesis)._
  _01/04/25._'
---

# How to use docker (and docker compose)?

## _A practical use tutorial_

_Sébastien Kerbourc'h_




<br/>
<br/>
<br/>
Source: mainly docs.docker.com

<!--
Adrien POIRE

Link to preopen:
- https://hub.docker.com/_/ros
- https://gitlab.ensta.fr/ssh/mirador/container_registry
- http://0.0.0.0:3000/
- konsole `docker compose up` & `docker exec -it rosbridge /bin/bash`

-->

---

## _A practical use tutorial_

<div class="center">
<img src="./docker_explaining.webp" height="400px" />
</div>

<!--

A **practical** use tutorial

-->


---

<!--
header: 'How to use docker? - **Table of contents**'
-->

1. What is docker
2. Dockerhub
3. Create images
4. Run an image
5. Publish images (Dockerhub & GitLab ENSTA)
6. To go deeper: Docker compose, useful run options (Volumes, X11, GPU)
7. A small example (Mirador + Fake robot)

---
<!--
header: 'How to use docker? - **What is docker**'
-->

<div class="center">
<img src="./docker_logo.png" height="100px" />
</div>

- Create isolated$^{*}$ _containers_ that bundles software programs/libraries/configurations
- Works at the OS Level
- Use kernel of the host ($\Rightarrow$ Difference from VM $\Rightarrow$ fewer resources )

<div class="center">
<a href="https://hub.docker.com/_/ros">
<img src="./works_on_my_machine.png" height="300px" />
</a>
</div>

---

- Image: The template/recipe for creating a container
- Container: A runnable instance of an image
- Registry: A repository of docker images

---

<!--
header: 'How to use docker? - **Dockerhub**'
-->

- Dockerhub (hub.docker.com): Main public registry provided by Docker, Inc.
  <br/>
  <br/>
  <br/>
  <br/>
  <br/>

---

- Dockerhub (hub.docker.com): Main public registry provided by Docker, Inc.
- ROS, ROS2
  <br/>
  <br/>
  <br/>
  <br/>

---

- Dockerhub (hub.docker.com): Main public registry provided by Docker, Inc.
- ROS, ROS2, Coq
  <br/>
  <br/>
  <br/>
  <br/>

---

- Dockerhub (hub.docker.com): Main public registry provided by Docker, Inc.
- ROS, ROS2, Coq, NodeJS
  <br/>
  <br/>
  <br/>
  <br/>

---

- Dockerhub (hub.docker.com): Main public registry provided by Docker, Inc.
- ROS, ROS2, Coq, NodeJS, argent/argent
  <br/>
  <br/>
  <br/>
  <br/>

---

- Dockerhub (hub.docker.com): Main public registry provided by Docker, Inc.
- ROS, ROS2, Coq, NodeJS, argent/argent, Docker

<div class="center">
<img src="./docker_in_docker.png" height="200px" />
</div>

---

- Dockerhub (hub.docker.com): Main public registry provided by Docker, Inc.
- ROS, ROS2, Coq, NodeJS, argent/argent, Docker, Debian, TensorFlow, PostgreSQL, MySQL, Ubuntu, etc

<!--
show dockerhub

-->
<div class="center">
<a href="https://hub.docker.com/_/ros">
<img src="./ros_dockerhub.png" height="200px" />
</a>
</div>


---

- Get an image using `docker pull <image name>[:<image tag>]` (default `<image tag>` is `latest`)
- Get argent/argent image `docker pull argent/argent`

<div class="center">
<img src="./docker_pull.png" height="100px" />
</div>

- List locally available images `docker images`

<div class="center">
<img src="./docker_images.png" height="100px" />
</div>

---

<!--
header: 'How to use docker? - **Create Images**'
-->

Main supported instructions (5/17):

- `FROM`: Declare the base image
- `COPY`: Copy files from host into the container
- `RUN`: Execute build commands
- `ENTRYPOINT`/`CMD`: Specify default executable/commands

```dockerfile
FROM node:23-slim

WORKDIR /app/
RUN apt update && apt install git -y &&\
    git clone -b refactor https://gitlab.ensta.fr/ssh/mirador /app &&\
    npm install
EXPOSE 3000

CMD ["node", "src/mirador.js"]
```

- `docker build -t <image name>[:<image tag>] .`

<!--
ENTRYPOINT = binary to start (e.g `/bin/sh -c`)
CMD = the arg given to the binary (e.g `bash`)
ADD	Add local or remote files and directories.
ARG	Use build-time variables.
COPY	Copy files and directories.
ENV	Set environment variables.
EXPOSE	Describe which ports your application is listening on.
FROM	Create a new build stage from a base image.
HEALTHCHECK	Check a container's health on startup.
LABEL	Add metadata to an image.
ONBUILD	Specify instructions for when the image is used in a build.
RUN	Execute build commands.
SHELL	Set the default shell of an image.
STOPSIGNAL	Specify the system call signal for exiting a container.
USER	Set user and group ID.
VOLUME	Create volume mounts.
WORKDIR	Change working directory.
-->

---

- `docker build -t registry-gitlab.ensta.fr:443/ssh/mirador .`

<div class="center">
<img src="./docker_build.png" height="200px" />
</div>

<div class="center">
<img src="./docker_building.png" height="200px" />
</div>

<!--

https://gitlab.ensta.fr/ssh/mirador/container_registry

-->


---

<!--
header: 'How to use docker? - **Run an Image**'
-->

- `docker run [options] <image name>[:<image tag>] [CMD]`: start a new container
- `docker run registry-gitlab.ensta.fr:443/ssh/mirador`

<div class="center">
<img src="./docker_run.png" height="70px" />
</div>

- `docker ps [-a]`: list running/existing container

<div class="center">
<img src="./docker_ps.png" height="70px" />
</div>

- `docker stop <image name>[:<image tag>]`, `docker start <image name>[:<image tag>]`,
  `docker rm <image name>[:<image tag>]`: start/stop/remove an existing container

---

<!--
header: 'How to use docker? - **Publish images (Dockerhub & GitLab ENSTA)**'
-->

- `docker push <image name>[:<image tag>]`: Send the docker image to the registry


- Default registry is Dockerhub (hub.docker.com).
- GitLab Registry available on gitlab.ensta.fr! (Deploy > Container Registry)

- `docker push registry-gitlab.ensta.fr:443/ssh/mirador`

<div class="center">
<img src="./docker_push.png" height="200px" />
</div>

---

<!--
header: 'How to use docker? - **Docker Compose**'
-->

- Usually one docker for each software (e.g. Frontend NodeJS + Backend NodeJS + MySQL)
- Docker compose: multiple `docker run` $\rightarrow$ JSON config (and more)

<div class="center">
<img src="./docker_everywhere.jpg" height="200px" />
</div>

---

<!--
header: 'How to use docker? - **Useful run options**'
-->

- `-i -t`: Keeps STDIN open, Allocates a pseudo-tty
- `--rm`: remove automatically the container after it stops
- `-v <mount path on the host path or name>:<mount path in the container>`: mount a host folder or a persistent volume
- `--gpu all` (requires `nvidia-container-toolkit`): make the gpu available
- `--device /dev/dri:/dev/dri --ipc=host -e DISPLAY=$DISPLAY
  -e XAUTHORITY=/root/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix`: make X11 server (GUI) available <!--
 -->(like ssh X11 forwarding)

---

<!--
header: 'How to use docker? - **A Small Example: Test Mirador app**'
-->

<div class="center">
<img src="./mermaid.png" height="400px" />
</div>

<!--
flowchart LR
    A[Thibault] <-.->|http| B(Mirador backend)
    C[Clément] <-.->|http| B(Mirador backend)
    D[Adrien] <-.->|http| B(Mirador backend)
    E[Alexandre] <-.->|http| B(Mirador backend)
    B(Mirador backend) <-.->|web socket| T1D([ROS Bridge])
    B(Mirador backend) <-.->|web socket| T2D([ROS Bridge])
    B(Mirador backend) <-.->|web socket| B1D([ROS Bridge])
    B(Mirador backend) <-.->|web socket| B2D([ROS Bridge])

    subgraph T1 [Tundra 1]
        T1D <-.-> ROST1([ROS])
    end
    subgraph T2 [Tundra 2]
        T2D <-.-> ROST2([ROS])
    end
    subgraph B1 [Baracuda 1]
        B1D <-.-> ROSB1([ROS])
    end
    subgraph B2 [Baracuda 2]
        B2D <-.-> ROSB2([ROS])
    end
-->

---

### Mirador

```dockerfile
FROM node:23-slim

WORKDIR /app/
RUN apt update && apt install git -y &&\
    git clone -b refactor https://gitlab.ensta.fr/ssh/mirador /app &&\
    npm install
EXPOSE 3000

CMD ["node", "src/mirador.js"]
```


---

### ROS Bridge

```dockerfile
FROM ros:noetic-robot

WORKDIR /catkin_ws
SHELL ["/bin/bash", "-c"]

RUN apt update &&\
    apt install -y git python3-catkin-tools python3-osrf-pycommon &&\
    apt install -y ros-noetic-rosbridge-server ros-noetic-geographic-msgs ros-move-base-msgs \
                   ros-noetic-move-base libgeographic-dev geographiclib-tools &&\
    wget https://sourceforge.net/projects/geographiclib/files/distrib-C++/GeographicLib-2.3.tar.gz && \
    tar -xvzf GeographicLib-2.3.tar.gz && \
    cd GeographicLib-2.3 && \
    mkdir BUILD && \
    cd BUILD && \
    cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++ && \
    make && make test && make install &&\
    mkdir /catkin_ws/src && cd /catkin_ws/src && git clone https://gitlab.ensta.fr/ssh/mirador_driver.git &&\
    cd /catkin_ws &&\
    . /opt/ros/noetic/setup.bash &&\
    catkin build mirador_driver

ENTRYPOINT ["/bin/bash"]
CMD [ "-c", "source /catkin_ws/devel/setup.bash && roslaunch rosbridge_server rosbridge_websocket.launch" ]
```

---

### Docker compose

```yaml
services:
  mirador:
    build:
      context: .
      dockerfile: Dockerfile
    image: registry-gitlab.ensta.fr:443/ssh/mirador
    container_name: mirador
    ports:
      - "3000:3000"
    volumes:
      - "./state.json:/app/state.json"
    networks:
      - shared_network
  
  rosbridge:
    build:
      context: .
      dockerfile: rosbridge.Dockerfile
    image: registry-gitlab.ensta.fr:443/ssh/mirador/rosbridge
    container_name: rosbridge
    stdin_open: true
    networks:
      - shared_network

networks:
  shared_network:
```

---

`docker compose up [-d]`

<!--
source devel/setup.bash

rostopic pub /mirador/status mirador_driver/Status "signal_quality: 50
pose: {latitude: 48.71085683066172, longitude: 2.2180272508537557, altitude: 70.0, heading: 0.0}
mode: 0
mission:
  header:
    seq: 0
    stamp: {secs: 0, nsecs: 0}
    frame_id: ''
  id: ''
  type: 0
  points:
  - {latitude: 0.0, longitude: 0.0, altitude: 0.0}
is_running: false
state_of_charge: 0
flight_status: 0
camera_elevation: 0.0
camera_zoom: 0
e_stop: false
stream_method: 0
stream_address: ['']"
-->

---

<!--
header: 'How to use docker?'
-->

<div class="center">

\*
\*&nbsp;&nbsp;&nbsp;\*
</div>
