# Dual-BEV Nav: Dual-layer BEV-based Heuristic Path Planning for Robotic Navigation in Unstructured Outdoor Environments

Link https://arxiv.org/pdf/2501.18351

LBPM Local BEV plannning model = local BEV perception encoder + task-driven goal decoder

## LBPM

### Local BEV Perception encoder

inputs: (i) context observation $o_{t-P:t-1}$ (ii) current observation $o_t$.

- BEV transformation on the observation
- Feature extract based on LSS method. Uses EfficientNet
- Uses LSS and BEVDet to predict discrete depth distribution for each pixel
- If multiple feature -> BEV pooling using BEVFusion

### Task-driven goal decoder

TODO