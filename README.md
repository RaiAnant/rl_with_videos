# RLMU


The baseline implementation is based on [softlearning](https://github.com/rail-berkeley/softlearning).

Experiment results are here: [link](https://drive.google.com/drive/folders/19qjlR42c-1s-S--AjPLdIxRzvsyuJ_FZ?usp=sharing)


# Installation

We recommend making a new virtual environment to install the dependencies.

```
git clone https://github.com/kschmeckpeper/rl_with_videos.git
cd rl_with_videos

pip install -r requirements.txt
python setup.py develop
```


# Examples
We provide the commands to replicate experiments from the paper.

## Acrobot
We wrap the Acrobot environment in the AcrobotContinuous environment, which takes a continuous action and discretizes it before passing it to the original Acrobot environment.

To run the SAC baseline, run the following commands.
```
cd examples/run_rl
python3 -u main.py --task=AcrobotContinuous-v1 --algorithm SAC --exp-name EXP_NAME --gpus=1 --trial-gpus=1
```


To run RLV, first download a replay pool containing the desired observations.  You may also use a replay pool generated during the training of SAC.

| Avg. Reward | Link |
| :----- | :----- |
| -99 | [here](https://drive.google.com/file/d/16Je5LcjTM_7VJ4oEjxNrAyoXhFdzHrwT/view?usp=sharing) |
| -79 | [here](https://drive.google.com/file/d/10v6TPAjQEoj9649qx4b76siwzTNwGL-8/view?usp=sharing) |
| -63 | [here](https://drive.google.com/file/d/15pqxuLvD-PjkWsdZl2FRyekpOhC88LPE/view?usp=sharing) |

Then, run the following commands.

```
cd examples/run_rl
python -u main.py --task=AcrobotContinuous-v1 --algorithm RLV --exp-name EXP_NAME --replace_rewards_bottom=-1.0 --replace_rewards_scale=10.0 --gpus=1 --trial-gpus=1 --replay_pool_load_path PATH/TO/REPLAY/POOL
```

```
python -u main.py --task=AcrobotContinuous-v1 --algorithm RLVU --exp-name EXP_NAME --replace_rewards_bottom=-1.0 --replace_rewards_scale=10.0 --gpus=1 --trial-gpus=1 --replay_pool_load_path C:\nyu\DRL\final_project\dataset\acrobot-975-1000.pkl --video_data_path C:\nyu\DRL\final_project\dataset\acrobot_sequence.pkl
```

## Pushing with Human Observations

To run the SAC baseline, run the following command.

```
cd examples/run_rl
python3 -u main.py --task=Image48HumanLikeSawyerPushForwardEnv-v0 --domain mujoco --algorithm SAC --exp-name EXP_NAME --gpus=1 --trial-gpus=1
```

To run RLV, first download the human observations from [here](https://drive.google.com/file/d/1osBobdYMwlqSEs5UA6UXzpiCPlbm1FNR/view?usp=sharing) and the human paired data from [here](https://drive.google.com/file/d/1qK2EoHMaOPAmACIxLbxI0C34gyH_UiWB/view?usp=sharing).

Run the following commands:

```
cd examples/run_rl
python3 -u main.py --task=Image48HumanLikeSawyerPushForwardEnv-v0 --domain mujoco --algorithm RLV  --exp-name EXP_NAME --gpus=1 --trial-gpus=1 --replay_pool_load_path /PATH/TO/REPLAY/POOL --paired_data_path /PATH/TO/PAIRED/DATA --paired_loss_scale 1e-06 --replace_rewards_scale=10.0 --replace_rewards_bottom=0.0 --domain_shift --domain_shift_generator_weight 0.001 --domain_shift_discriminator_weight 1e-08
```

To run the RV with video understanding, run the following command.

```
cd examples/run_rl
python -u main.py --task=Image48HumanLikeSawyerPushForwardEnv-v0 --domain mujoco --algorithm RLVU  --exp-name EXP_NAME --gpus=2 --trial-gpus=1 --replay_pool_load_path /PATH/TO/REPLAY/POOL --paired_data_path /PATH/TO/PAIRED/DATA --paired_loss_scale 1e-06 --replace_rewards_scale=10.0 --replace_rewards_bottom=0.0 --domain_shift --domain_shift_generator_weight 0.001 --domain_shift_discriminator_weight 1e-08
```


## Drawer opening with Human Observations

To run the SAC baseline, run the following command.

```
cd examples/run_rl
python3 -u main.py --task=Image48MetaworldDrawerOpenSparse2D-v0 --domain Metaworld --algorithm SAC  --exp-name EXP_NAME --gpus=1 --trial-gpus=1
```


Download the human observations from [here](https://drive.google.com/file/d/1LhJ5LE8FkiBI9i7KRtv-wFmsN5xvLkzu/view?usp=sharing) and the human paired data from [here](https://drive.google.com/file/d/1z-4XJevn-S2yHf8usk2Cv_NfzlT-pTYQ/view?usp=sharing).

Run the following commands:
```
cd examples/run_rl
python3 -u main.py --task=Image48MetaworldDrawerOpenSparse2D-v0 --domain Metaworld --algorithm RLV --exp-name EXP_NAME --gpus=1 --trial-gpus=1 --replay_pool_load_path /PATH/TO/REPLAY/POOL --paired_data_path /PATH/TO/PAIRED/DATA --paired_loss_scale 1e-08 --replace_rewards_scale=10.0 --replace_rewards_bottom=0.0 --domain_shift --domain_shift_generator_weight 0.001 --domain_shift_discriminator_weight 1e-08
```




## Citation
If this codebase helps you in your academic research, you are encouraged to cite our paper. Here is an example bibtex:
```
@article{schmeckpeper2020rlv,
  title={Reinforcement Learning with Videos: Combining Offline Observations with Interaction},
  author={Schmeckpeper, Karl and Rybkin, Oleh and Daniilidis, Kostas and Levine, Sergey and Finn, Chelsea},
  journal={Conference on Robot Learning},
  year={2020}
}
```



