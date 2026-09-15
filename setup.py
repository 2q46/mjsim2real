from setuptools import setup, find_packages

setup(
    name="pickcube-mjsim2real-rl",
    version="0.1.0",
    packages=find_packages(
        include=["rl", "rl.*", "env", "env.*", "utility", "utility.*"]
    ),
    python_requires=">=3.11",
    install_requires=[
        "mediapy",
        "wandb",
        "mujoco",
        "warp-lang==1.16.0",
        "mujoco-warp==3.11.0",
        "flax",
        "optax",
        "distrax",
        "orbax"
    ]
)