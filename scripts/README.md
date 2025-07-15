Python scripts to test/automate things
===

This README only discuss the testing scripts

## Run scripts

For scripts not depending on ros, it is possible (and easy) to use
python with docker. A docker image is still available if needed.

**For scripts depending on ros, it is recommended to use the [docker image](../docker/README.md) provided here.**
Alternatively, you can install ROS noetic on your machine and run the scripts directly.

> [!NOTE]
> If a script needs ROS to run, it will import the module scripts.rosutils. This will raise an ImportError if ROS is not
> detected (using the ROS_DISTRO env var).

### Env installation

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
```

### Run scripts

```sh
# (after sourcing the virtual environment)
python -m scripts.a_script
```

## List of scripts

- Bird-eye view
    - [rgbd_zed.py](bev/rgbd_zed.py)
    - [rgb_warping_zed.py](bev/rgb_warping_zed.py)
- Terrain segmentation
    - [detectree_test.py](terrain_segmentation/detectree_test.py)
    - [get_ign_map.py](terrain_segmentation/get_ign_map.py)
    - [detectree_on_ign_map.py](terrain_segmentation/detectree_on_ign_map.py)
    - [fast_colortexture_seg_outdoor_robots.py](terrain_segmentation/fast_colortexture_seg_outdoor_robots.py)
    - [segment_anything_test.py](terrain_segmentation/segment_anything_test.py)
    - [u_net_segmentation_test.py](terrain_segmentation/u_net_segmentation_test.py)
- ROS Related
    - [rosbag.py](rosutils/rosbag.py)
- Utils
    - [datasets.py](utils/datasets.py)
    - [monitoring.py](utils/monitoring.py)
    - [plot.py](utils/plot.py)
