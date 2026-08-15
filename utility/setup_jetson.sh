docker run --runtime nvidia -it --rm \
  --network host \
  --ipc=host \
  -v ~/.jax_cache:/tmp/jax_cache \
  -v ~/.warp_cache:/tmp/warp_cache \
  -v /home/ishan/mjsim2real:/mjsim2real \
  -w /mjsim2real \
  -e XLA_FLAGS="--xla_gpu_autotune_level=0" \
  -e JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache \
  -e WARP_CACHE_DIR=/tmp/warp_cache \
  mjsim2real:v1