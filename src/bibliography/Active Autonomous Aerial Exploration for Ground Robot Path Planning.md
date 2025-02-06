Active Autonomous Aerial Exploration for Ground Robot Path Planning
===

> [!NOTE]
> https://ieeexplore.ieee.org/document/7812671

1. Initial manual fly to see goal. Camera imagery is also used to obtain initial classification of the terrain.
2. Then vision-guided flights (to a series of waypoints) chosen actively. For each waypoint 3D reconstruction and ground
   robot path (to optimize total duration of the mission).
3. Repeat 2. until path is complete


- visual odometry for loc&nav [^3], [^4]
- [^3] also seems to mention matching of FPV (ground drone) and BEV (aerial drone) (p191)
- classifier is trained on the spot [^1]. Two models type tried, feature-based and CNN. CNN is significantly slower
  without giving significant better result (surprising).
- in the 2. not exhaustive exploration just along the _global path_ from the manually generated map.
  The next waypoints for the aerial drone are chosen as to minimize
  $T_{\text{ground robot}, s \rightarrow b_i} + T_{\text{ground robot}, b_i \rightarrow g} + T_{extend 3d reconstructed region}$,
  respectively _time of ground robot from s to next ground robot waypoint (uses 3d reconstructed area)_, the _time of
  ground robot from next ground robot waypoint (uses initial partial map)_, the _time to extend the 3d reconstructed
  region (in the correct direction)_.
- 7,8,9 are ref for high altitude, high resolution aerial images
- use of monocular camera to reconstruct 3D ground in real time [^2] (could be used with move_base_flex).
- use of [ANYbotics Grid Map](https://github.com/ANYbotics/grid_map) [^5]
- Terrain classification, dense 3D reconstruction and exploration algorithm run on a laptop computer not on the drone.

[^1]: https://rpg.ifi.uzh.ch/docs/ISER16_Delmerico.pdf

[^2]: https://rpg.ifi.uzh.ch/docs/ICRA14_Pizzoli.pdf

[^3]: https://rpg.ifi.uzh.ch/docs/PhD16_Forster.pdf

[^4]: https://ieeexplore.ieee.org/document/6906584

[^5]: https://www.researchgate.net/publication/284415855_A_Universal_Grid_Map_Library_Implementation_and_Use_Case_for_Rough_Terrain_Navigation