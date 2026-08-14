echo "nameserver 1.1.1.1" > /etc/resolv.conf
echo "nameserver 8.8.8.8" >> /etc/resolv.conf

# 2. Clear the broken index URL environment variable
unset PIP_INDEX_URL
unset PIP_EXTRA_INDEX_URL

# 3. Install MuJoCo as a single single-line command
pip install mujoco mujoco-warp warp-lang --index-url https://pypi.org/simple

pip install msgpack toolz dm-tree opt_einsum typing_extensions --index-url https://pypi.org/simple

pip install --no-deps chex flax optax distrax --index-url https://pypi.org/simple

python3 -c "
import jax
import flax
import optax
import distrax

print('JAX Devices:', jax.devices())
print('Flax version:', flax.__version__)
print('Optax version:', optax.__version__)
print('Distrax version:', distrax.__version__)
"