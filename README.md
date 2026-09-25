# LepidIO: Learning-Based Phase-Informed Dynamic Inertial Odometry for Flapping-Wing Robots

**ICRA Submission — Paper Under Review**

------

## 📄 About This Repository

Thank you for your interest in our work on **LepidIO**, a learning-based phase-informed inertial odometry framework tailored for flapping-wing robots.

**This repository is currently being prepared for public release.** We are committed to making our code, models, and data available to the community. The release is planned **upon acceptance of the paper currently under review**. We kindly ask reviewers and visitors to check back soon. 

------

## 📌 Paper Summary

We introduce **LepidIO**, a novel inertial odometry method tailored for flapping-wing robots. State estimation for bio-inspired flapping-wing micro aerial vehicles (FWMAVs) is challenged by body motion that differs fundamentally from the high-frequency vibration commonly observed on multirotors. Particularly, low-frequency wing strokes induce large, phase-locked body undulations and pitch oscillations. These motions obscure the relationship between inertial measurements and vehicle displacement, while attitude errors can amplify gravity-projection errors during inertial propagation.

**LepidIO** addresses these challenges through:

- A **causal Temporal Convolutional Network (TCN)** and **Gated Recurrent Unit (GRU)** that process synchronized gyroscope and accelerometer measurements
- Integration of **commanded wing-stroke phases** and **measured flapping-angle feedback**
- Prediction of **next-step three-dimensional relative displacement** in the body frame

We evaluate LepidIO on **26 real-flight sequences** collected using a butterfly-inspired FWMAV flapping at approximately 3 Hz, with an ultra-wideband (UWB) positioning system providing reference trajectories. Experimental results demonstrate that LepidIO localizes flapping-wing flight more accurately than representative methods under identical evaluation conditions. Our ablation study confirms that wing-stroke phase and flapping angle provide useful and complementary cues for localization.

------

## 🚀 Planned Release Contents

Upon paper acceptance, we will release the following:

| Resource                        | Description                                           |
| :------------------------------ | :---------------------------------------------------- |
| **Network Architecture Config** | Configuration file for the TCN-GRU model              |
| **Pretrained Network Weights**  | Trained model weights for reproducing paper results   |
| **Dataset**                     | Real-flight sequences with UWB reference trajectories |
| **Reproduction Scripts**        | Scripts to reproduce evaluation results               |

------

## ⭐ Stay Updated

Watch this repository to be notified when the code is released. We look forward to sharing our work with the community!
