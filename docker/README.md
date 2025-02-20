Docker images
===

To simplify reproducibility, a collection of Docker images are available to build/test/code.

> [!NOTE]
> If you don't want to use/have Docker, I recommend using the dockerfiles as a reference to install the dependencies.

| Image          | Description                                        | Dockerfile                             |
|----------------|----------------------------------------------------|----------------------------------------|
| sebsti1/latex  | A LaTeX environment to build the latex files       | [latex.Dockerfile](latex.Dockerfile)   |
| sebsti1/python | A Python environment to run the scripts            | [python.Dockerfile](python.Dockerfile) |
| sebsti1/ros    | A ROS environment to run the ROS tools and scripts | [ros.Dockerfile](ros.Dockerfile)       |

> [!IMPORTANT]
> The images should be built from the root folder of the repository (meaning not from the `docker` folder).

> [!NOTE]
> The images are also available on my [Docker Hub](https://hub.docker.com/u/sebsti1) page.

## Latex

| Command                                                                              | Description                            |
|--------------------------------------------------------------------------------------|----------------------------------------|
| `docker build -t sebsti1/mt_latex -f docker/latex.Dockerfile .`                      | Build the image                        |
| `docker run -v /path/to/mastersthesis:/app/ --name latex --rm sebsti1/mt_latex make` | Run the image to build the latex files |

## Python

| Command                                                                               | Description                      |
|---------------------------------------------------------------------------------------|----------------------------------|
| `docker build -t sebsti1/mt_python -f docker/python.Dockerfile .`                     | Build the image                  |
| `docker run -v /path/to/mastersthesis:/app/ --name python -it --rm sebsti1/mt_python` | Run the image to run the scripts |

## ROS

| Command                                                                         | Description                      |
|---------------------------------------------------------------------------------|----------------------------------|
| `docker build -t sebsti1/mt_ros -f docker/ros.Dockerfile .`                     | Build the image                  |
| `docker run -v /path/to/mastersthesis:/app/ --name ros -it --rm sebsti1/mt_ros` | Run the image to run the scripts |