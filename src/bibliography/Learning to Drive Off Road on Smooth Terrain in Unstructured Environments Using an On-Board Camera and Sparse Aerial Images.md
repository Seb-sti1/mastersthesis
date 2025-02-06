Learning to Drive Off Road on Smooth Terrain in Unstructured Environments Using an On-Board Camera and Sparse Aerial Images
===

> [!NOTE]
> https://arxiv.org/pdf/2004.04697


> In this paper, we present a system for learning a navigation
> policy that **preferentially chooses smooth terrain** [...]. The
> emphasis of the paper, however, is not road classification per
> se, but rather to propose an approach for **online adaptive self-
> supervised learning** for off-road driving in rough terrain and
> to explore the synthesis of aerial and first-person (ground)
> sensing in this context.

- From BEV, FPS images, use CNN to predict the rougher terrain for $H$ steps (supplying an action for every step)
- use of Value Prediction Networks, a hybrid model-based and model-free reinforcement learning architecture.
    - > It is model-based as it implicitly learns a dynamics model for abstract states optimized for predicting future
      rewards and value functions.
    - > It is also model-free as it maps these encoded abstract states to rewards and value functions using direct
      experience with the environment prior to the planning phase.
- In training terrain roughness is estimated using IMU and obstacles using short-range LiDAR.
- Reward based on the difference between the prediction and the actual roughness
- It is classification as type of terrain is associated with different number and the model tries to predict the correct
  one


