UAV-Assisted Self-Supervised Terrain Awareness for Off-Road Navigation
===

> [!NOTE]
> https://arxiv.org/pdf/2409.18253

- good documentation of related work
- datasets created could be very useful
- makes me think of the work done by Tom.

use of ResNet18 and a "homemade" MLP to predict $M_z$, $M_\omega$ and $M_p$
respectively vibration metric, bumpiness and electrical energy consumption.

BEV from UAV gives better prediction than FPS from UGV.

Using the BEV from UAV, can obtain maps (similar to occupancy grid) (see Fig5)
than can be used to choose "better" path.

