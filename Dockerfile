FROM dustynv/jax:r36.3.0-cu126

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENV JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache \
    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0

WORKDIR /mjsim2real

RUN pip install --no-cache-dir mujoco warp-lang mujoco-warp --index-url https://pypi.org/simple

RUN pip install --no-cache-dir \
    msgpack \
    toolz \
    dm-tree \
    opt_einsum \
    typing_extensions \
    mediapy \
    wandb \
    tensorflow_probability --index-url https://pypi.org/simple

RUN pip install --no-cache-dir --no-deps \
    chex \
    flax \
    optax \
    distrax --index-url https://pypi.org/simple


RUN mkdir -p /tmp/jax_cache && chmod 777 /tmp/jax_cache

CMD ["/bin/bash"]
