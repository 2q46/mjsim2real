import jax
import time
import optax
import jax.numpy as jnp
from rl.ppo_cfg import ActorConfig, CriticConfig
from rl.ppo_rgb import \
    ActorNetwork, CriticNetwork, compute_actor_loss, compute_critic_loss, compute_advantage_estimates
from env.mjenv import \
    init_mujoco, init_rendering, sample_action, step_batch, render_batch, reset_batch, find_goal_cube_pos


def main(
    num_envs: int, 
    eps: float, 
    lambda_: float, 
    gamma_: float, 
    num_epochs: int, 
    lr: int, 
    checkpoint_freq: int, 
    image_res: tuple,
    n_timesteps: int,
    seed: int
):

    mj_model, mjw_model, mjw_data = init_mujoco(num_envs)
    render_ctx, rgb_buff = init_rendering(mj_model, num_envs, image_res)
    main_rng_key = jax.random.PRNGKey(seed)
    epoch_rngs = jax.random.split(main_rng_key, num=num_epochs)
    actor_rng, critic_rng = jax.random.split(main_rng_key, num=2)
    actor_cfg = ActorConfig(img_size=image_res)
    critic_cfg = CriticConfig(img_size=image_res)

    actor_network = ActorNetwork(cfg=actor_cfg)
    critic_network = CriticNetwork(cfg=critic_cfg)
    dummy_obs = jnp.zeros((num_envs, *image_res, 3), dtype=jnp.float32)
    actor_params = actor_network.init(actor_rng, dummy_obs)
    critic_params = critic_network.init(critic_rng, dummy_obs)
    tx = optax.chain(
      optax.clip_by_global_norm(0.5),
      optax.adam(learning_rate=lr, eps=1e-5),
    ) 
    global_step = 0

    for epoch in range(1, num_epochs+1):
        obs_buffer, actions_buff, log_prob_buff = [], [], []
        value_buff, reward_buff = [], []
        reset_batch(mjw_model, mjw_data, epoch_rngs[epoch])
        goal_cube_pos = find_goal_cube_pos(mj_model, mjw_data)
        obs = render_batch(mjw_model, mjw_data, render_ctx, rgb_buff)
        timesteps_rng = jax.random.split(main_rng_key, num=n_timesteps)

        for i in range(n_timesteps):
            global_step += num_envs
            policy = actor_network(obs)
            value = critic_network(obs)
            action, log_prob = policy.sample_and_log_prob(timesteps_rng[n_timesteps-1])
            rew = step_batch(mj_model, mjw_model, mjw_data, action, goal_cube_pos)
            obs = render_batch(mjw_model, mjw_data, render_ctx, rgb_buff)
            reward_buff.append(rew)
            log_prob_buff.append(log_prob.sum(dim=1))
            obs_buffer.append(obs)
            actions_buff.append(action)
            value_buff.append(value)

        obs_arr = jnp.stack(obs_buffer)
        actions_arr = jnp.stack(actions_buff)
        log_prob_arr = jnp.stack(log_prob_buff)
        value_arr = jnp.stack(value_buff)
        reward_arr = jnp.stack(reward_arr)

        print(obs_arr.shape)
        print(actions_arr.shape)
        print(log_prob_arr.shape)
        print(value_arr.shape)
        print(reward_arr.shape)
        return None
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--gamma_", type=float, default=0.99)
    parser.add_argument("--num_epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_freq", type=int, default=10)
    parser.add_argument("--image_res", type=tuple, default=(128, 128))
    parser.add_argument("--n_timesteps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if len(jax.devices("cuda")) > 0:
        print("CUDA-capable device available...")
        main(
            args.num_envs, 
            args.eps, 
            args.lambda_, 
            args.gamma_, 
            args.num_epochs, 
            args.lr, 
            args.checkpoint_freq, 
            args.image_res,
            args.n_timesteps,
            args.seed
        )
    else:
        print("No CUDA-capable device available.")