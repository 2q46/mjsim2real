import jax
from distrax import MultivariateNormalDiag
import jax.numpy as jnp
import flax.linen as nn
from ppo_cfg import ActorConfig, CriticConfig

class ActorNetwork(nn.Module):

    cfg: ActorConfig

    @nn.compact
    def __call__(self, x):
        dtype = jnp.float32
        x = x.astype(dtype) / 255.0
        n_batches = x.shape[0]
        log_std = self.param("log_std", nn.initializers.constant(self.cfg.start_log_std), (n_batches, self.cfg.output_features))
        x = nn.Conv(features=self.cfg.features[0], kernel_size=self.cfg.kernel_size, dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.cfg.dropout_rate)(x)
        x = nn.Conv(features=self.cfg.features[1], kernel_size=self.cfg.kernel_size, dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.cfg.dropout_rate)(x)
        x = nn.Conv(features=self.cfg.features[2], kernel_size=self.cfg.kernel_size, dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.cfg.dropout_rate)(x)
        x = nn.Conv(features=self.cfg.features[3], kernel_size=self.cfg.kernel_size, dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.cfg.dropout_rate)(x)
        x = nn.Conv(features=self.cfg.features[4], kernel_size=self.cfg.kernel_size, dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.cfg.dropout_rate)(x)
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
        x = 2*nn.tanh(x)
        return MultivariateNormalDiag(x, jnp.exp(log_std))

class CriticNetwork(nn.Module):

    cfg: CriticConfig

    @nn.compact
    def __call__(self, x):
        dtype = jnp.float32
        x = x.astype(dtype) / 255.0
        x = nn.Conv(features=self.cfg.features[0], kernel_size=self.cfg.kernel_size, dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.cfg.dropout_rate)(x)
        x = nn.Conv(features=self.cfg.features[1], kernel_size=self.cfg.kernel_size, dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.cfg.dropout_rate)(x)
        x = nn.Conv(features=self.cfg.features[2], kernel_size=self.cfg.kernel_size, dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.cfg.dropout_rate)(x)
        x = nn.Conv(features=self.cfg.features[3], kernel_size=self.cfg.kernel_size, dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.cfg.dropout_rate)(x)
        x = nn.Conv(features=self.cfg.features[4], kernel_size=self.cfg.kernel_size, dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.cfg.dropout_rate)(x)
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
        x = nn.relu(x)
        return x

@jax.jit(static_argnums=[1])
def compute_rew_to_go(episode_rew: jax.Array, gamma_: float=0.99):

    timefirst_ep_rew = jnp.moveaxis(episode_rew, 1, 0)
    initial_carry = jnp.zeros_like(timefirst_ep_rew[-1, :])

    def update_previous(prev_rew, xs):
        current_rew = xs + prev_rew * gamma_
        return current_rew

    last_carry, timefirst_discounted_rew = jax.lax.scan(
        update_previous, 
        init=initial_carry, 
        xs=timefirst_ep_rew,
        reverse=True
    )
    batchfirst_discounted_rew = jnp.moveaxis(timefirst_discounted_rew, 0, 1)
    return batchfirst_discounted_rew

@jax.jit(static_argnums=[2,3])
def compute_advantage_estimates(
        state_value_estimates: jax.Array,
        episode_rew: jax.Array,
        gamma_: float=0.99,
        lambda_: float=0.95
    ): 
    n_batches, n_timesteps = episode_rew.shape[0], episode_rew.shape[1] - 1
    gae_last = episode_rew[:, n_timesteps] - state_value_estimates[:, n_timesteps]
    tfirst_v_t = jnp.moveaxis(state_value_estimates, 1, 0)
    tfirst_v_next = jnp.concat([tfirst_v_t[1:], jnp.zeros_like((1, n_batches))], axis=0)
    tfirst_episode_rew = jnp.moveaxis(episode_rew, 1, 0)
    
    def compute_prev_gae(gae_next, xs): 
        v_next, v_t, r_t = xs
        delta = gamma_ * v_next + r_t - v_t
        gae_now = delta + lambda_ * gamma_ * gae_next
        return gae_now, gae_now

    last_carry, gae = jax.lax.scan(
        compute_prev_gae, 
        init=gae_last, 
        xs=(tfirst_v_next, tfirst_v_t, tfirst_episode_rew), 
        reverse=True
    )
    gae_batch_first = jnp.moveaxis(gae, 0, 1)
    return gae_batch_first
    
@jax.jit
def compute_critic_loss(estimated_rew_to_go: jax.Array, gt_rew_to_go: jax.Array):
    squared_err = jnp.linalg.norm(gt_rew_to_go - estimated_rew_to_go, axis=1)
    mean_squared_err = jnp.mean(squared_err)
    return mean_squared_err
    
@jax.jit(static_argnums=[3])
def compute_actor_loss(new_log_prob: jax.Array, old_log_prob: jax.Array, advantage_func: jax.Array, eps=0.05):
    ratio = jnp.exp(new_log_prob - old_log_prob)
    cliped_ratio = jnp.clip(ratio, min=1-eps, max=1+eps)
    surr1 = cliped_ratio * advantage_func
    surr2 = ratio * advantage_func
    loss = jnp.minimum(surr1, surr2).mean()
    return -loss
