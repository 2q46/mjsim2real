import jax
from distrax import MultivariateNormalDiag
import jax.numpy as jnp
import flax.linen as nn
import flax.struct as struct
from typing import Tuple
from functools import partial

@struct.dataclass
class ActorConfig:

    img_size: int = 128
    output_features: int = 6
    features: Tuple[int, ...] = (24, 16, 16, 4, 1)
    dense_features: Tuple[int, ...] = (128, 128, 32, 16)
    kernel_size: tuple = (3, 3)
    dropout_rate: float = 5e-2
    start_log_std: float = -0.1


@struct.dataclass
class CriticConfig:

    img_size: int = 128
    features: Tuple[int, ...] = (24, 16, 16, 4, 1)
    dense_features: Tuple[int, ...] =(128, 64, 32, 16)
    kernel_size: tuple = (3, 3)
    dropout_rate: float = 5e-2


class ActorNetwork(nn.Module):

    cfg: ActorConfig

    @nn.compact
    def __call__(self, x):
        dtype = jnp.float32
        x = x.astype(dtype) / 255.0
        x = (x - 0.5) / 0.5
        log_std = self.param("log_std", nn.initializers.constant(self.cfg.start_log_std), (1, self.cfg.output_features))
        x = nn.Conv(features=self.cfg.features[0], strides=(2, 2), kernel_size=self.cfg.kernel_size, dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.cfg.dropout_rate, deterministic=True)(x)
        x = nn.Conv(features=self.cfg.features[1], strides=(2, 2), kernel_size=self.cfg.kernel_size, dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.cfg.dropout_rate, deterministic=True)(x)
        x = nn.Conv(features=self.cfg.features[2], strides=(2, 2), kernel_size=self.cfg.kernel_size, dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.cfg.dropout_rate, deterministic=True)(x)
        x = nn.Conv(features=self.cfg.features[3], kernel_size=self.cfg.kernel_size, dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.cfg.dropout_rate, deterministic=True)(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.cfg.dense_features[0], dtype=dtype)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.cfg.dense_features[1], dtype=dtype)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.cfg.dense_features[2], dtype=dtype)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.cfg.dense_features[3], dtype=dtype)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.cfg.output_features, dtype=dtype)(x)
        x = 1.5 * nn.tanh(x)
        return MultivariateNormalDiag(x, jnp.exp(log_std))

class CriticNetwork(nn.Module):

    cfg: CriticConfig

    @nn.compact
    def __call__(self, x):
        dtype = jnp.float32
        x = x.astype(dtype) / 255.0
        x = (x - 0.5) / 0.5
        x = nn.Conv(features=self.cfg.features[0], strides=(2, 2), kernel_size=self.cfg.kernel_size, dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.cfg.dropout_rate, deterministic=True)(x)
        x = nn.Conv(features=self.cfg.features[1], strides=(2, 2), kernel_size=self.cfg.kernel_size, dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.cfg.dropout_rate, deterministic=True)(x)
        x = nn.Conv(features=self.cfg.features[2], strides=(2, 2), kernel_size=self.cfg.kernel_size, dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.cfg.dropout_rate, deterministic=True)(x)
        x = nn.Conv(features=self.cfg.features[3], kernel_size=self.cfg.kernel_size, dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.cfg.dropout_rate, deterministic=True)(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.cfg.dense_features[0], dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(self.cfg.dense_features[1], dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(self.cfg.dense_features[2], dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(self.cfg.dense_features[3], dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(1, dtype=dtype)(x)
        return x.squeeze(-1)

@partial(jax.jit, static_argnames=["gamma_"])
def compute_rew_to_go(episode_rew: jax.Array, gamma_: float=0.99):

    mean_episode_rew = jnp.mean(jnp.sum(episode_rew, axis=1))
    timefirst_ep_rew = jnp.moveaxis(episode_rew, 1, 0)
    initial_carry = jnp.zeros_like(timefirst_ep_rew[-1, :])

    def update_previous(prev_rew, xs):
        current_rew = xs + prev_rew * gamma_
        return current_rew, current_rew

    last_carry, timefirst_discounted_rew = jax.lax.scan(
        update_previous, 
        init=initial_carry, 
        xs=timefirst_ep_rew,
        reverse=True
    )
    #timefirst_discounted_rew = jnp.flip(timefirst_discounted_rew, axis=0)
    batchfirst_discounted_rew = jnp.moveaxis(timefirst_discounted_rew, 0, 1)
    return batchfirst_discounted_rew, mean_episode_rew

@partial(jax.jit)
def compute_adv_estimates( # Q - V
    state_value_estimates: jax.Array,
    gt_rew_to_go: jax.Array
):

    adv_estimates = gt_rew_to_go - state_value_estimates
    adv_estimates = (adv_estimates - adv_estimates.mean()) / (1e-8 + adv_estimates.std())
    return adv_estimates

@partial(jax.jit, static_argnames=["gamma_", "lambda_"])
def compute_advantage_estimates(
    state_value_estimates: jax.Array,  # Shape: (num_envs, n_timesteps)
    episode_rew: jax.Array,            # Shape: (num_envs, n_timesteps)
    gamma_: float = 0.99,
    lambda_: float = 0.95,
) -> jax.Array:
    num_envs = episode_rew.shape[0]

    # Move time dimension to axis 0: (n_timesteps, num_envs)
    v_t = jnp.moveaxis(state_value_estimates, 1, 0)
    r_t = jnp.moveaxis(episode_rew, 1, 0)

    # Scan step: carry holds (gae_next, v_next)
    def compute_prev_gae(carry, xs):
        gae_next, v_next = carry
        v_curr, r_curr = xs

        delta = r_curr + gamma_ * v_next - v_curr
        gae_curr = delta + (gamma_ * lambda_) * gae_next

        # New carry is (gae_curr, v_curr) for step t-1
        return (gae_curr, v_curr), gae_curr

    # Boundary conditions at T: gae_T = 0, v_T = 0 (or boot-strapped value)
    init_carry = (jnp.zeros(num_envs), jnp.zeros(num_envs))

    # Scan backward from T-1 to 0
    _, gae = jax.lax.scan(
        compute_prev_gae,
        init=init_carry,
        xs=(v_t, r_t),
        reverse=True
    )

    # Transpose back to (num_envs, n_timesteps)
    gae_batch_first = jnp.moveaxis(gae, 0, 1)

    # Normalize advantages
    gae_batch_first = (gae_batch_first - gae_batch_first.mean()) / (
        gae_batch_first.std() + 1e-8
    )

    return gae_batch_first

@partial(jax.jit)
def compute_eval_metrics(
    is_success_buffer: jax.Array, 
    is_touching_buffer: jax.Array,
    is_grasped_buffer: jax.Array
    ):
    num_success = jnp.sum(jnp.any(is_success_buffer, axis=1))
    num_touching = jnp.sum(jnp.any(is_touching_buffer, axis=1))
    num_is_grasped = jnp.sum(jnp.any(is_grasped_buffer, axis=1))
    return num_success, num_touching, num_is_grasped