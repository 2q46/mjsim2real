echo "nameserver 1.1.1.1" > /etc/resolv.conf
echo "nameserver 8.8.8.8" >> /etc/resolv.conf

# 2. Clear the broken index URL environment variable
unset PIP_INDEX_URL
unset PIP_EXTRA_INDEX_URL

# 3. Install MuJoCo as a single single-line command
pip install mujoco mujoco-warp warp-lang optax flax distrax --index-url https://pypi.org/simple
