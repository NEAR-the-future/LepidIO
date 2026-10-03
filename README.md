# LepidIO: Learning-Based Phase-Informed Dynamic Inertial Odometry for Flapping-Wing Robots

**ICRA Submission — Paper Under Review**

------

## About

Thank you for your interest in our work on **LepidIO**! LepidIO is a learning-based phase-informed inertial odometry framework tailored for flapping-wing robots. It combines gyroscope and accelerometer measurements with **commanded wing-stroke phase** and **measured flapping-angle** feedback, using a causal **Temporal Convolutional Network (TCN)** and a **Gated Recurrent Unit (GRU)** to predict **next-step three-dimensional displacement** in the body frame.

The method is designed to account for the **pronounced, phase-dependent body motions** of flapping-wing flight. The paper evaluates LepidIO on **26 real-flight sequences** collected using a butterfly-inspired robot, with ultra-wideband (UWB) positioning providing reference trajectories.

------

## Current Release

This repository provides an **initial evaluation release** for reviewers and interested visitors. It includes the model implementation and supporting utilities needed to evaluate the pretrained model on **two released flight sequences**.

| Resource                         | Location                                                     | Description                                                  |
| :------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Evaluation data                  | `data/20260903_171951.csv`, `data/20260903_191508.csv`       | Two flight sequences listed in the checkpoint's test split.  |
| Pretrained weights               | `results/lepidio_c5/ckpt/best_model.pt`                      | Released LepidIO checkpoint.                                 |
| Network configuration            | `configs/lepid_io.json`                                      | Configuration for the released model and inference pipeline. |
| Model and split metadata         | `results/lepidio_c5/ckpt/model_param.json`, `results/lepidio_c5/ckpt/data_split.json` | Saved model parameters and sequence split information.       |
| Inference and evaluation scripts | `tools/network_only_trajectory.py`, `tools/odometry_evaluation.py` | Trajectory reconstruction, metric calculation, and visualization. |

The two released sequences are a subset of the data used in the paper. This release supports evaluation of the provided checkpoint; it is not yet a complete package for reproducing every experiment. 

------

## Evaluate the Pretrained Model

Download the repository and run the following commands from its root directory, preferably in a virtual environment.

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Reconstruct and evaluate trajectories for both released sequences:

```bash
for sequence in 20260903_171951 20260903_191508; do
  MPLBACKEND=Agg python tools/network_only_trajectory.py \
    --config configs/lepid_io.json \
    --checkpoint results/lepidio_c5/ckpt/best_model.pt \
    --csv "data/${sequence}.csv" \
    --out-dir "results/evaluation/${sequence}" \
    --device cpu

  python tools/odometry_evaluation.py evaluate \
    --sequence "${sequence}" \
    --gt "data/${sequence}.csv" \
    --traj "results/evaluation/${sequence}/stamped_traj_network.txt" \
    --config configs/lepid_io.json \
    --checkpoint results/lepidio_c5/ckpt/best_model.pt \
    --network-source released_checkpoint \
    --out-dir "results/evaluation/${sequence}" \
    --device cpu
done
```

Use `--device cuda:0` instead of `--device cpu` if a compatible CUDA-enabled PyTorch installation is available.

Outputs are saved under `results/evaluation/<sequence>/`, including the reconstructed trajectory, displacement predictions, JSON/CSV metrics, and trajectory and error plots. A combined summary can be generated with:

```bash
python tools/odometry_evaluation.py summarize \
  --run-dir results/evaluation \
  --sequences 20260903_171951 20260903_191508
```

### Evaluation Scope

The released configuration uses a one-second input window at 100 Hz. Trajectory reconstruction rotates predicted body-frame displacements into the world frame using the recorded CSV attitude quaternions. Reference positions initialize the first input window; subsequent positions are accumulated from network predictions without further position-reference updates. The trajectory file retains the recorded attitude, so its orientation columns do not represent a learned attitude estimate.

The position RMSE in `network_trajectory_metrics.json` excludes the initialization window. The additional trajectory statistics from `odometry_evaluation.py` use the full associated trajectory, including initialization; these metrics therefore have different evaluation intervals.

------

## Planned Full Release

The **full project release is planned upon acceptance of the paper**, including the complete codebase and accompanying data and documentation. This README will be updated with the release contents and usage instructions.

Thank you for your interest in LepidIO. Please revisit this repository for the full release and future updates.
