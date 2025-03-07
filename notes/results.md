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