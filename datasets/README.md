Datasets sources
===

Below is a list of datasets (not present in this is repository) that were used to tests different method/algorithms.

| Folder name            | Link                                   | Additional comment                              |
|------------------------|----------------------------------------|-------------------------------------------------|
| aukerman               | https://hub.dronedb.app/r/odm/aukerman |                                                 |
| rosbag_u2is            |                                        | See [next section](#internal-dataset-locations) |
| viewpro_calibration    |                                        | See [next section](#internal-dataset-locations) |
| norlab_ulaval_datasets | https://arxiv.org/abs/2409.18253       | Not publicly available                          |

## Internal dataset locations

The following datasets are from U2IS lab and are not publicly available. It is list the relevant information to find
them aim at my colleagues.

| File name                                                   | sha1sum                                  | Hard drive type      | sha1sum of SN                            | Additional comment |
|-------------------------------------------------------------|------------------------------------------|----------------------|------------------------------------------|--------------------|
| rosbag_u2is/ENSTA_U2IS_grass_2024-05-03-14-54-33.bag        | 024bbbf7366e47f98bdf0e567576ea95c2d59b1d | Transcend 4To (blue) | ed09a74906fd5d559bfee3acf3d0706f649b24ff |                    |
| rosbag_u2is/ENSTA_U2IS_road_shadows_2024-05-22-16-33-03.bag | e589e67c5ca7f69d4cf927fe150b60b01573ba50 | Transcend 4To (blue) | ed09a74906fd5d559bfee3acf3d0706f649b24ff |                    |
| rosbag_u2is/pal_diff_ang_2024-04-18-14-07-54.bag            | 2639f29b4e458da5997966bcbd1ea5bbd3973893 | Transcend 4To (blue) | ed09a74906fd5d559bfee3acf3d0706f649b24ff |                    |
| viewpro_calibration                                         | NA                                       | NA                   | NA                                       |                    |

_See [find_bag.sh](find_bag.sh) for a script using the provided information to find the bag file._