import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.struct as struct
from typing import Tuple
from functools import partial
import distrax


@struct.dataclass
class SACActorConfig:
    img_size: int = 128
    output_features: int = 6
    features: Tuple[int, ...] = (16, 16, 8, 4)
    dense_features: Tuple[int, ...] = (256, 256, 128, 32)
    log_std_features: Tuple[int, ...] = (64, 32)
    kernel_size: tuple = (3, 3)
    dropout_rate: float = 5e-2
    log_std_min: float = -5.0
    log_std_max: float = 2.0


@struct.dataclass
class SACCriticConfig:
    img_size: int = 128
    features: Tuple[int, ...] = (16, 16, 8, 4)
    dense_features: Tuple[int, ...] = (256, 256, 128, 16)
    kernel_size: tuple = (3, 3)
    dropout_rate: float = 5e-2


def _conv_trunk(x, cfg, dtype):
    x = x.astype(dtype) / 255.0
    x = (x - 0.5) / 0.5
    x = nn.Conv(features=cfg.features[0], strides=(4, 4), kernel_size=cfg.kernel_size, dtype=dtype)(x)
    x = nn.relu(x)
    x = nn.Dropout(cfg.dropout_rate, deterministic=True)(x)
    x = nn.Conv(features=cfg.features[1], strides=(2, 2), kernel_size=cfg.kernel_size, dtype=dtype)(x)
    x = nn.relu(x)
    x = nn.Dropout(cfg.dropout_rate, deterministic=True)(x)
    x = nn.Conv(features=cfg.features[2], strides=(2, 2), kernel_size=cfg.kernel_size, dtype=dtype)(x)
    x = nn.relu(x)
    x = nn.Dropout(cfg.dropout_rate, deterministic=True)(x)
    x = nn.Conv(features=cfg.features[3], kernel_size=cfg.kernel_size, dtype=dtype)(x)
    x = nn.relu(x)
    x = nn.Dropout(cfg.dropout_rate, deterministic=True)(x)
    return x.reshape((x.shape[0], -1))


class SACActorNetwork(nn.Module):
    """Squashed (tanh) diagonal-Gaussian policy. Returns a distrax.Transformed
    distribution so `.sample_and_log_prob` already includes the tanh
    change-of-variables correction, matching the calling convention used by
    the PPO actor in rl/ppo_rgb.py."""

    cfg: SACActorConfig

    @nn.compact
    def __call__(self, x):
        dtype = jnp.float32
        x = _conv_trunk(x, self.cfg, dtype)

        h = nn.Dense(self.cfg.dense_features[0], dtype=dtype)(x)
        h = nn.relu(h)
        h = nn.Dense(self.cfg.dense_features[1], dtype=dtype)(h)
        h = nn.relu(h)
        h = nn.Dense(self.cfg.dense_features[2], dtype=dtype)(h)
        h = nn.relu(h)
        h = nn.Dense(self.cfg.dense_features[3], dtype=dtype)(h)
        h = nn.relu(h)

        mean = nn.Dense(self.cfg.output_features, dtype=dtype)(h)

        log_std = nn.Dense(self.cfg.log_std_features[0], dtype=dtype)(x)
        log_std = nn.tanh(log_std)
        log_std = nn.Dense(self.cfg.log_std_features[1], dtype=dtype)(log_std)
        log_std = nn.tanh(log_std)
        log_std = nn.Dense(self.cfg.output_features, dtype=dtype)(log_std)
        log_std = jnp.clip(log_std, self.cfg.log_std_min, self.cfg.log_std_max)

        base_dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        bijector = distrax.Block(distrax.Tanh(), ndims=1)
        return distrax.Transformed(base_dist, bijector)


class SACCriticNetwork(nn.Module):
    """Q(obs, action) -> scalar. Conv trunk on image obs, action concatenated
    in before the dense head."""

    cfg: SACCriticConfig

    @nn.compact
    def __call__(self, x, action):
        dtype = jnp.float32
        x = _conv_trunk(x, self.cfg, dtype)
        x = jnp.concatenate([x, action.astype(dtype)], axis=-1)

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


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

def compute_critic_loss(
    critic_params,
    critic_apply_fn,
    target_q,
    obs,
    action,
):
    q1, q2 = critic_apply_fn(critic_params, obs, action)
    loss = jnp.mean((q1 - target_q) ** 2) + jnp.mean((q2 - target_q) ** 2)
    return loss, (q1, q2)


def compute_actor_and_alpha_loss(
    actor_params,
    actor_apply_fn,
    critic_apply_fn,
    critic_params,
    log_alpha,
    obs,
    key,
    target_entropy,
):
    policy = actor_apply_fn(actor_params, obs)
    actions, log_prob = policy.sample_and_log_prob(seed=key)

    q1, q2 = critic_apply_fn(critic_params, obs, actions)
    q_min = jnp.minimum(q1, q2)

    alpha = jax.lax.stop_gradient(jnp.exp(log_alpha))
    actor_loss = jnp.mean(alpha * log_prob - q_min)

    # alpha is optimized separately (stop-gradient on log_prob here since we
    # only want d(alpha_loss)/d(log_alpha)); kept as its own function below.
    return actor_loss, log_prob


def compute_alpha_loss(log_alpha, log_prob, target_entropy):
    log_prob = jax.lax.stop_gradient(log_prob)
    return jnp.mean(-jnp.exp(log_alpha) * (log_prob + target_entropy))


# --------------------------------------------------------------------------
# Target computation + train steps
# --------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(1, 2))
def compute_target_q(
    actor_params,
    actor_apply_fn,
    critic_target_apply_fn,
    critic_target_params,
    log_alpha,
    next_obs,
    reward,
    done,
    gamma,
    key,
):
    next_policy = actor_apply_fn(actor_params, next_obs)
    next_action, next_log_prob = next_policy.sample_and_log_prob(seed=key)

    next_q1, next_q2 = critic_target_apply_fn(critic_target_params, next_obs, next_action)
    next_q_min = jnp.minimum(next_q1, next_q2)

    alpha = jnp.exp(log_alpha)
    next_v = next_q_min - alpha * next_log_prob
    target_q = reward + gamma * (1.0 - done) * next_v
    return jax.lax.stop_gradient(target_q)


@partial(jax.jit, static_argnums=(1, 2))
def critic_train_step(
    critic_params,
    critic_apply_fn,
    critic_tx_update,
    critic_opt_state,
    obs,
    action,
    target_q,
):
    (loss, (q1, q2)), grads = jax.value_and_grad(compute_critic_loss, has_aux=True)(
        critic_params, critic_apply_fn, target_q, obs, action
    )
    updates, new_opt_state = critic_tx_update(grads, critic_opt_state, critic_params)
    new_params = optax_apply_updates(critic_params, updates)
    return new_params, new_opt_state, loss, q1, q2


@partial(jax.jit, static_argnums=(1, 2, 4))
def actor_train_step(
    actor_params,
    actor_apply_fn,
    critic_apply_fn,
    critic_params,
    actor_tx_update,
    actor_opt_state,
    log_alpha,
    obs,
    key,
    target_entropy,
):
    (actor_loss, log_prob), grads = jax.value_and_grad(compute_actor_and_alpha_loss, has_aux=True)(
        actor_params, actor_apply_fn, critic_apply_fn, critic_params, log_alpha, obs, key, target_entropy
    )
    updates, new_opt_state = actor_tx_update(grads, actor_opt_state, actor_params)
    new_params = optax_apply_updates(actor_params, updates)
    return new_params, new_opt_state, actor_loss, log_prob


@partial(jax.jit, static_argnums=(1,))
def alpha_train_step(log_alpha, alpha_tx_update, alpha_opt_state, log_prob, target_entropy):
    alpha_loss, grad = jax.value_and_grad(compute_alpha_loss)(log_alpha, log_prob, target_entropy)
    updates, new_opt_state = alpha_tx_update(grad, alpha_opt_state, log_alpha)
    new_log_alpha = optax_apply_updates(log_alpha, updates)
    return new_log_alpha, new_opt_state, alpha_loss


@jax.jit
def soft_update(target_params, online_params, tau):
    return jax.tree_util.tree_map(
        lambda t, o: tau * o + (1.0 - tau) * t, target_params, online_params
    )


# small helper kept local so this file has no import-order issues with optax
import optax as _optax


def optax_apply_updates(params, updates):
    return _optax.apply_updates(params, updates)