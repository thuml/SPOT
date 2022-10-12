# Supported Policy Optimization

Official implementation for NeurIPS 2022 paper [Supported Policy Optimization for Offline Reinforcement Learning](https://arxiv.org/abs/2202.06239).

## Environment

1. Install [MuJoCo version 2.0](https://www.roboti.us/download.html) at ~/.mujoco/mujoco200 and copy license key to ~/.mujoco/mjkey.txt
2. Create a conda environment
```
conda env create -f conda_env.yml
conda activate spot
```
3. Install [D4RL](https://github.com/rail-berkeley/d4rl)

## Usage

### Pretrained Models

We have uploaded pretrained VAE models and offline models to facilitate experiment reproduction. Download from this [link](https://drive.google.com/file/d/1_v6yPpwYw6T7CcBs1u_UJizf9wZmV1PW/view?usp=sharing) and unzip:

```
unzip spot-models.zip -d .
```

### Offline RL

Run the following command to train VAE.

```
python train_vae.py --env halfcheetah --dataset medium-replay
python train_vae.py --env antmaze --dataset medium-diverse --no_normalize
```

Run the following command to train offline RL on D4RL with pretrained VAE models.

```
python main.py --config configs/offline/halfcheetah-medium-replay.yml
python main.py --config configs/offline/antmaze-medium-diverse.yml
```

You can also specify the random seed and VAE model:

```
python main.py --config configs/offline/halfcheetah-medium-replay.yml --seed <seed> --vae_model_path <vae_model.pt>
```

#### Logging

This codebase uses tensorboard. You can view saved runs with:

```
tensorboard --logdir <run_dir>
```

### Online Fine-tuning

Run the following command to online fine-tune on AntMaze with pretrained VAE models and offline models.

```
python main_finetune.py --config configs/online_finetune/antmaze-medium-diverse.yml
```

You can also specify the random seed, VAE model and offline models:

```
python main_finetune.py --config configs/online_finetune/antmaze-medium-diverse.yml --seed <seed> --vae_model_path <vae_model.pt> --pretrain_model <pretrain_model/>
```

## Citation

If you find this code useful for your research, please cite our paper as:

```
@inproceedings{wu2022supported,
  title={Supported Policy Optimization for Offline Reinforcement Learning},
  author={Jialong Wu and Haixu Wu and Zihan Qiu and Jianmin Wang and Mingsheng Long},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## Contact

If you have any question, please contact wujialong0229@gmail.com .

## Acknowledgement

This repo borrows heavily from [sfujim/TD3_BC](https://github.com/sfujim/TD3_BC) and [sfujim/BCQ](https://github.com/sfujim/BCQ).