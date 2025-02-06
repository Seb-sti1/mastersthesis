Notes
===

This is a markdown to organize the collection of my notes.

## Articles

- LSS https://arxiv.org/pdf/2008.05711
- ViKiNG https://arxiv.org/pdf/2202.11271  https://sites.google.com/view/viking-release
- https://arxiv.org/pdf/2004.04697 20 21 24 25

### With Helicopter

- For
  landing https://www.researchgate.net/publication/257343605_Autonomous_Landing_at_Unprepared_Sites_by_a_Full-Scale_Helicopter
- Helicopter is sending sensors (stereo cam & lidar) to the
  UGV https://www.researchgate.net/publication/344982070_Learning_to_Drive_Off_Road_on_Smooth_Terrain_in_Unstructured_Environments_Using_an_On-Board_Camera_and_Sparse_Aerial_Images

### mbf (move_base_flex) related

- https://ora.ox.ac.uk/objects/uuid:0758dad0-6b33-40d1-bade-0cc2eb6f989a
- https://rpg.ifi.uzh.ch/docs/ICRA14_Pizzoli.pdf

### Short descriptions

https://arxiv.org/pdf/2004.04697

#### A two-stage level set evolution scheme for man-made objects detection in aerial images

https://www.researchgate.net/publication/4156206_A_two-stage_level_set_evolution_scheme_for_man-made_objects_detection_in_aerial_images

Detect region in images of man-made object

#### Experimental Analysis of Overhead Data Processing To Support Long Range Navigation

(application to DARPA) https://ieeexplore.ieee.org/document/4058754
(classifier
description) https://kilthub.cmu.edu/articles/journal_contribution/Terrain_Classification_from_Aerial_Data_to_Support_Ground_Vehicle_Navigation/6561173?file=12043478

usage of LiDAR and imagery on (only) terrestrial robot and prior data (from variety of sources)
to achieve robust navigation.

#### 3D Convolutional Neural Networks for Landing Zone Detection from LiDAR

https://dimatura.net/publications/3dcnn_lz_maturana_scherer_icra15.pdf

Coupling of a volumetric occupancy map with a 3D CNN to distinguish between vegetation that may be landed on and solid
objects that should be avoided.

### Long descriptions

- [Active Autonomous Aerial Exploration for Ground Robot Path Planning](Active%20Autonomous%20Aerial%20Exploration%20for%20Ground%20Robot%20Path%20Planning.md)
- [Dual-BEV Nav: Dual-layer BEV-based Heuristic Path Planning for Robotic Navigation in Unstructured Outdoor Environments](Dual-BEV%20Nav%20Dual-layer%20BEV-based%20Heuristic%20Path%20Planning%20for%20Robotic%20Navigation%20in%20Unstructured%20Outdoor%20Environments.md)
- [UAV-Assisted Self-Supervised Terrain Awareness for Off-Road Navigation](UAV-Assisted%20Self-Supervised%20Terrain%20Awareness%20for%20Off-Road%20Navigation.md)
- [Visual Terrain Classification by Flying Robots](Visual%20Terrain%20Classification%20by%20Flying%20Robots.md)
- [Active Autonomous Aerial Exploration for Ground Robot Path Planning](Active%20Autonomous%20Aerial%20Exploration%20for%20Ground%20Robot%20Path%20Planning.md)

## Drone datasets & Maps

- UAV Datasets
    - https://github.com/OpenDroneMap/ODMdata
        - https://hub.dronedb.app/r/odm/aukerman
        - https://github.com/zivillian/odm_ziegeleipark/tree/master
        - https://drive.google.com/file/d/12UrLDHA6iZFJYF7OkErccH_xanRHefoZ/view
        - https://drive.google.com/file/d/1faBtGK7Jm5lTo_UWLz6onDGYGqlykHPa/view
        - https://drive.google.com/file/d/11yFommuRZyVXADcYEIjf-qz3AZrA8M6E/view
    - https://arxiv.org/pdf/2501.18351
- Camp de Beynes:
    - Geoportail :
      https://remonterletemps.ign.fr/telecharger/?lon=1.889434&lat=48.860337&z=13.4&layer=pva&couleur=C&year=2013
    - Google Maps : https://maps.app.goo.gl/zCesePWSyXjnszSf7

## Debug

makefile and latex https://tex.stackexchange.com/questions/40738/how-to-properly-make-a-latex-project

input vs include https://tex.stackexchange.com/questions/246/when-should-i-use-input-vs-include

jinja2 https://jinja.palletsprojects.com/en/stable/templates/ & https://github.com/pappasam/latexbuild

minted compilation issue https://github.com/gpoore/minted/issues/231

https://tex.stackexchange.com/questions/161094/adding-custom-metadata-values-to-a-pdf-file