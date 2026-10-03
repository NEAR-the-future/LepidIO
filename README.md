# LepidIO: Learning-Based Phase-Informed Dynamic Inertial Odometry for Flapping-Wing Robots

**ICRA Submission — Paper Under Review**

------

## 📄 About This Repository

Thank you for your interest in our work on **LepidIO**, a learning-based phase-informed inertial odometry framework tailored for flapping-wing robots.

## Version History

| Version | Updates |
| :------ | :------ |
| **v0.0.1** | Initial public release of the paper-aligned core pipeline, including the fixed network architecture, training and trajectory inference scripts, configuration, and pretrained model. |

------

## 📌 Paper Summary

We introduce **LepidIO**, a novel inertial odometry method tailored for flapping-wing robots. State estimation for bio-inspired flapping-wing micro aerial vehicles (FWMAVs) is challenged by body motion that differs fundamentally from the high-frequency vibration commonly observed on multirotors. Particularly, low-frequency wing strokes induce large, phase-locked body undulations and pitch oscillations. These motions obscure the relationship between inertial measurements and vehicle displacement, while attitude errors can amplify gravity-projection errors during inertial propagation.

**LepidIO** addresses these challenges through:

- A **causal Temporal Convolutional Network (TCN)** and **Gated Recurrent Unit (GRU)** that process synchronized gyroscope and accelerometer measurements
- Integration of **commanded wing-stroke phases** and **measured flapping-angle feedback**
- Prediction of **next-step three-dimensional relative displacement** in the body frame

We evaluate LepidIO on **26 real-flight sequences** collected using a butterfly-inspired FWMAV flapping at approximately 3 Hz, with an ultra-wideband (UWB) positioning system providing reference trajectories. Experimental results demonstrate that LepidIO localizes flapping-wing flight more accurately than representative methods under identical evaluation conditions. Our ablation study confirms that wing-stroke phase and flapping angle provide useful and complementary cues for localization.

------

## Training and Inference

Train the model using the default configuration:

```bash
python src/learning/main_net.py train \
  --config configs/lepid_io.json \
  --device cuda:0
```

Run network-only trajectory inference:

```bash
python tools/network_only_trajectory.py \
  --config configs/lepid_io.json \
  --checkpoint results/lepidio_c5/ckpt/best_model.pt \
  --csv datasets/Dataset_clean_raw_attitude/20260903_171951.csv \
  --out-dir results/lepidio_c5/20260903_171951 \
  --device cuda:0
```
