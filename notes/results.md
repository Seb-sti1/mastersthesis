Results of my research and tests
===

## IGN map

| Satellite image                    |
|------------------------------------|
| ![ign_map.jpg](images/ign_map.jpg) |

## Terrain segmentation

### Detectree

| 100px                                                            | 800px                                                            | 1000px                                                             |
|------------------------------------------------------------------|------------------------------------------------------------------|--------------------------------------------------------------------|
| ![detectree_100_DSC00229.png](images/detectree_100_DSC00229.png) | ![detectree_800_DSC00229.png](images/detectree_800_DSC00229.png) | ![detectree_1000_DSC00229.png](images/detectree_1000_DSC00229.png) |

![detectree_perf.svg](images/detectree_perf.svg)

| Satellite image                    | Satellite image classified                               |
|------------------------------------|----------------------------------------------------------|
| ![ign_map.jpg](images/ign_map.jpg) | ![ign_map_classified.jpg](images/ign_map_classified.jpg) |

## Calibration of the Viewpro Q10F

After some tests, at a given zoom, the calibration noise is bigger than the variation because of the
changes in focal length due to the focus (when adding gaussian noise (avg=0, std=1) on the coordinates
of the detected chessboard, the focal length changes by more than when only changing the focus setting)
Hence, a calibration matrix for each zoom seems reasonable.

[calibration.py](../scripts/utils/calibration.py) is used to perform calibrations and tests. For a zoom = 1, using
Kfold (5 folds) technique to evaluate the reprojection errors, it is close to 10.

![reprojecterror.png](images/calib_results/reprojecterror.png)

At zoom = 1,
```python
k = np.array([[1360, 0, 630], [0, 1360, 344], [0, 0, 1]])
d = np.array([[-1.60867089e-01], [3.00941722e-01],
              [5.51651036e-05], [6.23006522e-03],
              [0.00000000e+00]])
```

## BEV projection

### Using RGB, Depth and camera intrinsic parameters of the ZED

- Replay ROS bag using [rosbag.py](../../scripts/rosutils/rosbag.py).
- Topic of the zed `rgb/image_rect_color`, `depth/depth_registered` and `rgb/camera_info`
- For python implementation open3d doesn't help that much, using numba improve for-loop in bev significantly

| RGB                                                           | Depth                                                     | PCD                                                           |
|---------------------------------------------------------------|-----------------------------------------------------------|---------------------------------------------------------------|
| ![bev_zed_RGBD_RGB.png](images/bev_rgbd/bev_zed_RGBD_RGB.png) | ![bev_zed_RGBD_D.png](images/bev_rgbd/bev_zed_RGBD_D.png) | ![bev_zed_RGBD_pcd.png](images/bev_rgbd/bev_zed_RGBD_pcd.png) |

| Small square column to do the bev                                     | BEV (f78229f2) 0.1, avg aggregation                               | BEV (75ca91e5) 0.05, avg aggregation                                    |
|-----------------------------------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------------|
| ![bev_zed_RGBD_pcd_seg.png](images/bev_rgbd/bev_zed_RGBD_pcd_seg.png) | ![bev_zed_RGBD_bev.png](images/bev_rgbd/bev_zed_RGBD_bev_0.1.png) | ![bev_zed_RGBD_bev_0.05.png](images/bev_rgbd/bev_zed_RGBD_bev_0.05.png) |

![bev_zed_RGBD_compute_time.svg](images/bev_rgbd/bev_zed_RGBD_compute_time.svg)

Accumulating points doesn't really make it better... The computation start at ~ 1Hz and slowly slows down to 0.5Hz (
after 526 iterations) while the accumulated pcd grown from 100k to 450k. Even with the use of ICP, the pcd ends up
having shifts aberration, especially in the trees.

| Accumulated pcd                                                                 | BEV from accumulated pcd (77b3ec64) (0.5)                                                         |
|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| ![bev_zed_RGBD_merged_cloud.png](images/bev_rgbd/bev_zed_RGBD_merged_cloud.png) | ![bev_zed_RGBD_merged_cloud_bev_0.05.png](images/bev_rgbd/bev_zed_RGBD_merged_cloud_bev_0.05.png) |

```mermaid
flowchart LR
    OldPCD[Old PCD]
    NewPCD[New PCD]
    NewPCD --> ExcludeTooFar1[Exclude points too far from center]
    OldPCD -->|Source| ICP
    ExcludeTooFar1 -->|Target| ICP
    ICP -->|Tf| Inverse[^-1]
    Inverse --> Transform
    OldPCD --> Transform
    Transform --> Merge
    ExcludeTooFar1 --> Merge
    Merge --> ExcludeTooFar2[Exclude points too far from center]
    ExcludeTooFar2 --> Resample[Voxel sample down - BEV_res/2]
    Resample --> NewOldPCD[New old PCD]
```

### Warping RGB ZED Image

See thesis report.

## UAV/UGV images pairing

### #1 Generating 6x2 grid of patch

I first tried to generate a 6x2 grid of patch in front of the uav and the ugv. The problem with this technic is that
first there is a lot of patch generated ($ \sim 6 \times 10^3$ for uav), the second is that there is even more
possible comparisons ($\mathcal{O}(n^2)$). My idea was to try to find if there was a correlation with the number of
matched points and distance between the position were the image was taken (one by the uav, the other with ugv).
Sadly, it didn't give any meaningful results: the variety of image was too big and the estimation of the distance
between two image was to approximate.

### #2 Test along the trajectory

After the first experiment, I decided to focus and benchmark different techniques for the matching. Using patches
extracted along the path of the robot oriented in the same direction as the one of the robot, I was able to
test Xfeat (with lighterglue, vanilla and star), omniglue and orb.

Using XFeat, it gives pretty promising results :

| UGV/UAV/UGV region in UAV                              | Keypoints matched                                            |
|--------------------------------------------------------|--------------------------------------------------------------|
| ![x_feat_found1.png](images/matching/test2/x_feat_found1.png) | ![x_feat_found1_kp.png](images/matching/test2/x_feat_found1_kp.png) |
| ![x_feat_found2.png](images/matching/test2/x_feat_found2.png) | ![x_feat_found2_kp.png](images/matching/test2/x_feat_found2_kp.png) |
| ![x_feat_found3.png](images/matching/test2/x_feat_found3.png) | ![x_feat_found3_kp.png](images/matching/test2/x_feat_found3_kp.png) |

A (really) small statistic analysis shows that there is a small correlation between the mean/max/min scores (of the
keypoints) and the number of matched keypoints. This means that implementing a strategy when taking picture of
key location that only keeps "good" images is something that can be considered.

| hist of number of matches                                                    | max of keypoints scores depending on the number of matches                         | mean of keypoints scores depending on the number of matches                          | min of keypoints scores depending on the number of matches                         |
|------------------------------------------------------------------------------|------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| ![x_feat_histogram_matches.svg](images/matching/test2/x_feat_histogram_matches.svg) | ![x_feat_keypoints_max_scores.svg](images/matching/test2/x_feat_keypoints_max_scores.svg) | ![x_feat_keypoints_mean_scores.svg](images/matching/test2/x_feat_keypoints_mean_scores.svg) | ![x_feat_keypoints_min_scores.svg](images/matching/test2/x_feat_keypoints_min_scores.svg) |

Using omniglue gives disappointing results. Even if it, sometimes, is able to find the correct location, it
has a had tendency to generate homography that do not represent something that is physically possible given
the constraints of the problem at hand (e.g. the reprojected image of the ugv in the uav is a crossed quadrilateral,
a quadrilateral where 3 points are close to aligned or an unreasonably small quadrilateral)

| UGV/UAV/UGV region in UAV                                  | Keypoints matched                                                |
|------------------------------------------------------------|------------------------------------------------------------------|
| ![omniglue_found1.png](images/matching/test2/omniglue_found1.png) | ![omniglue_found1_kp.png](images/matching/test2/omniglue_found1_kp.png) |

Side note: in order to do the inference, the image were all _resized_ to a 400x400 image. This can impact the results of
the algorithm but given how bad they are, I did not try to go in more details. Also, feasibility under onboard &
realtime constraints is also part of the "scoring" of the different methods.

### #3 0°, 90°, 180°, 270°

After these easy tests, I increased the complexity by adding rotation (multiple of 90°, in order to prevent any
deformation of the pixels at a micro level). This is because in the real experiment, the ugv will not be present
in the image and the orientation of it is _apriori_ unknown, therefore the relative orientation of the uav image
and ugv image is "random" (more of that in the next experiment).

Before re-running the benchmark, I add few indicator that help find obvious bad matches (e.g. the reprojected image of
the ugv in the uav is a crossed quadrilateral, etc.). I also normalised the images before applying the mean square
error (mse) and the structural similarity index measure (ssim) scores.

Sadly it gave pretty bad results, as even xfeat gave poor results when not on the 0° images.

<!-- no image because would need to be regenerated using 4b62d8c6 -->

A good way to see this is to look at the rate of obvious wrong images (see below). It clearly shows that when the angle
is not 0° the number of obvious wrong doubles (or even nearly triples).

| xfeat+ligherglue                                                                                        | xfeat                                                                         | xfeat$^*$                                                                                 | omniglue                                                                            | orb                                                                       |
|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| ![obviously_wrong_xfeat+lighterglue.png](images/matching/test3/obviously_wrong_xfeat%2Blighterglue.png) | ![obviously_wrong_xfeat.png](images/matching/test3/obviously_wrong_xfeat.png) | ![obviously_wrong_xfeat+star.png](images/matching/test3/obviously_wrong_xfeat%2Bstar.png) | ![obviously_wrong_omniglue.png](images/matching/test3/obviously_wrong_omniglue.png) | ![obviously_wrong_orb.png](images/matching/test3/obviously_wrong_orb.png) |

When only looking at the not obvious wrong images, the mse and ssim of the common region in the uav and ugv images
detected by the algorithms. It is important to keep in mind that the scores are really noisy and that the plot should
be used to understand the underlying trend. The first information shown in the plot is the repartition of the mse (lower
is better) and the ssim (higher is better) for each angle (when "mentally" ignoring the x-axis). It confirms that
even among not obvious wrong images when the rotation between images is not 0°, all the different algorithms give
pretty bad result. The second is the correlation with the number of found matches and the improvement in the scores.
These results also confirm that the xfeat+ligherglue algorithm is the best version at finding matches.

| algorithm        | 0°                                                                                        | 90 °                                                                                        | 180 °                                                                                         | 270 °                                                                                         |
|------------------|-------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| xfeat+ligherglue | ![0_scores_xfeat+lighterglue.png](images/matching/test3/0_scores_xfeat%2Blighterglue.png) | ![90_scores_xfeat+lighterglue.png](images/matching/test3/90_scores_xfeat%2Blighterglue.png) | ![180_scores_xfeat+lighterglue.png](images/matching/test3/180_scores_xfeat%2Blighterglue.png) | ![270_scores_xfeat+lighterglue.png](images/matching/test3/270_scores_xfeat%2Blighterglue.png) |
| xfeat            | ![0_scores_xfeat.png](images/matching/test3/0_scores_xfeat.png)                           | ![90_scores_xfeat.png](images/matching/test3/90_scores_xfeat.png)                           | ![180_scores_xfeat.png](images/matching/test3/180_scores_xfeat.png)                           | ![270_scores_xfeat.png](images/matching/test3/270_scores_xfeat.png)                           |
| xfeat$^*$        | ![0_scores_xfeat+star.png](images/matching/test3/0_scores_xfeat%2Bstar.png)               | ![90_scores_xfeat+star.png](images/matching/test3/90_scores_xfeat%2Bstar.png)               | ![180_scores_xfeat+star.png](images/matching/test3/180_scores_xfeat%2Bstar.png)               | ![270_scores_xfeat+star.png](images/matching/test3/270_scores_xfeat%2Bstar.png)               |
| omniglue         | ![0_scores_omniglue.png](images/matching/test3/0_scores_omniglue.png)                     | ![90_scores_omniglue.png](images/matching/test3/90_scores_omniglue.png)                     | ![180_scores_omniglue.png](images/matching/test3/180_scores_omniglue.png)                     | ![270_scores_omniglue.png](images/matching/test3/270_scores_omniglue.png)                     |
| orb              | ![0_scores_orb.png](images/matching/test3/0_scores_orb.png)                               | ![90_scores_orb.png](images/matching/test3/90_scores_orb.png)                               | ![180_scores_orb.png](images/matching/test3/180_scores_orb.png)                               | ![270_scores_orb.png](images/matching/test3/270_scores_orb.png)                               |

### #4 -10°, -5°, 0, 5°, 10°

In the previous section, it was assumed that there was no knowledge about the relative orientation between the
uav and the ugv image. This is partially wrong, as the robots measure their heading. Therefore, it is theoretically
possible to realign the images. For this new test, the dataset was regenerated in order to have uav images with a
relative angle of -10, -5, 0, 5, 10 degrees and verify that in this case the algorithms would be able to find the
regions. And, as shown in the next graph, it works great with xfeat+ligherglue!

| xfeat+ligherglue                                                                                        | xfeat                                                                         | xfeat$^*$                                                                                 | omniglue                                                                            | orb                                                                       |
|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| ![obviously_wrong_xfeat+lighterglue.png](images/matching/test4/obviously_wrong_xfeat%2Blighterglue.png) | ![obviously_wrong_xfeat.png](images/matching/test4/obviously_wrong_xfeat.png) | ![obviously_wrong_xfeat+star.png](images/matching/test4/obviously_wrong_xfeat%2Bstar.png) | ![obviously_wrong_omniglue.png](images/matching/test4/obviously_wrong_omniglue.png) | ![obviously_wrong_orb.png](images/matching/test4/obviously_wrong_orb.png) |

| algorithm        | -10°                                                                                          | -5 °                                                                                        | 0 °                                                                                       | 5 °                                                                                       | 10 °                                                                                        |
|------------------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| xfeat+ligherglue | ![-10_scores_xfeat+lighterglue.png](images/matching/test4/-10_scores_xfeat%2Blighterglue.png) | ![-5_scores_xfeat+lighterglue.png](images/matching/test4/-5_scores_xfeat%2Blighterglue.png) | ![0_scores_xfeat+lighterglue.png](images/matching/test4/0_scores_xfeat%2Blighterglue.png) | ![5_scores_xfeat+lighterglue.png](images/matching/test4/5_scores_xfeat%2Blighterglue.png) | ![10_scores_xfeat+lighterglue.png](images/matching/test4/10_scores_xfeat%2Blighterglue.png) |
| xfeat            | ![-10_scores_xfeat.png](images/matching/test4/-10_scores_xfeat.png)                           | ![-5_scores_xfeat.png](images/matching/test4/-5_scores_xfeat.png)                           | ![0_scores_xfeat.png](images/matching/test4/0_scores_xfeat.png)                           | ![5_scores_xfeat.png](images/matching/test4/5_scores_xfeat.png)                           | ![10_scores_xfeat.png](images/matching/test4/10_scores_xfeat.png)                           |
| xfeat$^*$        | ![-10_scores_xfeat+star.png](images/matching/test4/-10_scores_xfeat%2Bstar.png)               | ![-5_scores_xfeat+star.png](images/matching/test4/-5_scores_xfeat%2Bstar.png)               | ![0_scores_xfeat+star.png](images/matching/test4/0_scores_xfeat%2Bstar.png)               | ![5_scores_xfeat+star.png](images/matching/test4/5_scores_xfeat%2Bstar.png)               | ![10_scores_xfeat+star.png](images/matching/test4/10_scores_xfeat%2Bstar.png)               |
| omniglue         | ![-10_scores_omniglue.png](images/matching/test4/-10_scores_omniglue.png)                     | ![-5_scores_omniglue.png](images/matching/test4/-5_scores_omniglue.png)                     | ![0_scores_omniglue.png](images/matching/test4/0_scores_omniglue.png)                     | ![5_scores_omniglue.png](images/matching/test4/5_scores_omniglue.png)                     | ![10_scores_omniglue.png](images/matching/test4/10_scores_omniglue.png)                     |

Looking at the timings, it shows that xfeat+ligherglue (and other versions of xfeat) is faster than omniglue

| xfeat+ligherglue                                                                        | xfeat                                                           | xfeat$^*$                                                                   | omniglue                                                              | orb                                                         |
|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------------------|-----------------------------------------------------------------------|-------------------------------------------------------------|
| ![duration_xfeat+ligherglue.png](images/matching/test4/duration_xfeat%2Bligherglue.png) | ![duration_xfeat.png](images/matching/test4/duration_xfeat.png) | ![duration_xfeat+star.png](images/matching/test4/duration_xfeat%2Bstar.png) | ![duration_omniglue.png](images/matching/test4/duration_omniglue.png) | ![duration_orb.png](images/matching/test4/duration_orb.png) |

Seeing those results, confirms that xfeat+lighterglue is a good candidate.

### Simulation of the real algorithm

See thesis report.