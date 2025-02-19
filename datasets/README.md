Datasets sources
===

Below is a list of datasets (not present in this is repository) that were used to tests different method/algorithms.

| Folder name  | Link                                   | Additional comment                             |
|--------------|----------------------------------------|------------------------------------------------|
| aukerman     | https://hub.dronedb.app/r/odm/aukerman |                                                |
| rosbag_husky |                                        | See [next section](internal-dataset-locations) |

## Internal dataset locations

The following datasets are from U2IS lab and are not publicly available. It is list the relevant information to find
them aim at my colleagues.

| File name                                             | sha1sum                                  | Hard drive type      | sha1sum of SN                            | Additional comment |
|-------------------------------------------------------|------------------------------------------|----------------------|------------------------------------------|--------------------|
| rosbag_husky/ENSTA_U2IS_grass_2024-05-03-14-54-33.bag | 024bbbf7366e47f98bdf0e567576ea95c2d59b1d | Transcend 4To (blue) | ed09a74906fd5d559bfee3acf3d0706f649b24ff |                    |

_See [find_bag.sh](find_bag.sh) for a script using the provided information to find the bag file._
