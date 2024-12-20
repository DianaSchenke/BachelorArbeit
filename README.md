
## Package Requirements

The following packages should be installed as exactly the listed version:

>- pytorch 2.4.0 for cuda 11.8
>- accelerate 0.34.2
>- unsloth build 2024.11.7 and it's requirements; make sure to install the version matching the pytorch version
>- transformers 4.46.2 
>- trl 0.13.0 (the PPO script specifically requires trl==0.8.6 instead, which requires downgrading transformers and some other dependencies)

The remaining package requirements should be fairly version agnostic and should just be installed as needed using pip. A full list of packages (copy pasted from pip freeze) can be found in packages.md.

---

## Project Structure

The project is organized into the following folders:

>### base_datasets:
>All datasets are stored in this folder, with small scripts to generate them as needed.


>### dpo-orpo, ppo, sft:
>Training setups for DPO, ORPO, PPO and SFT training. **dataset_files** contains dataset files that are adapted to fit the input format of the various training methods as well as scripts to create them.
>The **loop** scripts contain a fairly generic training loop, which is started from a **start_script** scripts, which contains additional parameters and initializes the dataset (with exception of ppo, which is started through ppo_loop.py) .


>### evaluation_scripts:
>Contains various scripts that start various experiments to evaluate trained models using the evaluator class in **evaluator.py**. The results are stored as a pickled pandas dataframe in the results folder. 


>### output: 
>Trained models, model checkpoints and Tensorboard data is stored here. 


>### plots: 
>Contains various scripts that help produce the plots for my thesis using matplotlib.


>### utils:
>Various shared utilities I wrote that are used by scripts in multiple places.

---


## Using Accelerate

All scripts that involve training or inference on a model are designed to be run through accelerate. To do so, navigate to the script folder and execute the following command:

> accelerate launch *script_name*.py --config_file=accel_config.yaml

All folders containing scripts that need accelerate already contain a accel_config.yaml file, wich can be edited through a text editor to adjust accelerate parameters. For more detail please consult the official accelerate documentation.

## Using Tensorboard

All training scripts generate tensorboard data by default, which can be used to track training progress. Tensorboard can be viewed using the following command, executed from the base project folder:

> tensorboard --logdir output --bindall

For more information consult the official tensorboard documentation.

