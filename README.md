# Sim2real using Mujoco-Warp, LeRobot and SO101 Robot.

## Requirements:

- OS: Ubuntu 20.04 / 22.04 / 24.04 LTS (recommended)
- Python: 3.11
- GPU: NVIDIA GPU with VRAM ≥ 8 GB (for parallel GPU vectorization)

## Setup Training: 

For this project I would recommend training on Google Colab using a L4 GPU or a high RAM T4 GPU. I have perpared a notebook here: https://colab.research.google.com/drive/1v_svgTef-0FdNNdWHc2XNTXJSl5ECbSQ?authuser=0#scrollTo=aaahHedpbuhn. Once it has trained you can skip to the deployment steps. If you wish to run locally, follow the steps below. 

### 1. Setup a Conda environment

```bash
conda env -n "mjsim2real" python=3.11
conda activate mjsim2real
```

### 2. Install JAX

install a version of JAX compatible with your CUDA version using Pip from here https://docs.jax.dev/en/latest/installation.html. You can check your CUDA version running:

```bash
nvidia-smi
```

### 3. Install other project dependencies

```bash
pip install -e .
```
### 4. Begin training

If running on a GPU with more than 16 GB of VRAM I would reccomend increasing the number of parallel environments through the num_envs flag to 256 or more. Also you can increase the learning rate as well.

```bash
python main.py 
```

## Setup Deployment: 



