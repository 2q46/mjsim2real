import jax
import time
import optax
import jax.numpy as jnp
from rl.ppo_rgb import \
    ActorNetwork, CriticNetwork, ActorConfig, CriticConfig, \
    compute_advantage_estimates, compute_rew_to_go
from env.mjenv import \
    init_mujoco, init_rendering, sample_action, step_batch, \
    render_batch, reset_batch, find_goal_cube_pos


def main(
    num_envs: int, 
    eps: float, 
    lambda_: float, 
    gamma_: float, 
    num_epochs: int, 
    lr: float, 
    checkpoint_freq: int, 
    image_res: tuple,
    n_timesteps: int,
    seed: int,
    n_mini_batches: int
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
    
    dummy_obs = jnp.zeros((1, *image_res, 3), dtype=jnp.float32)
    actor_params = actor_network.init(actor_rng, dummy_obs)
    critic_params = critic_network.init(critic_rng, dummy_obs)

    actor_optim = optax.chain(
      optax.clip_by_global_norm(0.5),
      optax.adam(learning_rate=lr, eps=1e-5),
    ) 
    critic_optim = optax.chain(
      optax.clip_by_global_norm(0.5),
      optax.adam(learning_rate=lr, eps=1e-5),
    ) 

    actor_opt_state = actor_optim.init(actor_params)
    critic_opt_state = critic_optim.init(critic_params)

    @jax.jit
    def get_policy(actor_params, obs):
        return actor_network.apply(actor_params, obs)

    @jax.jit
    def get_value_estimates(critic_params, obs):
        return critic_network.apply(critic_params, obs)

    def compute_critic_loss(critic_params, obs, gt_rew_to_go: jax.Array):
        estimated_rew_to_go = critic_network.apply(critic_params, obs).squeeze(-1)
        squared_err = jnp.square(gt_rew_to_go - estimated_rew_to_go)
        return jnp.mean(squared_err)
    
    def compute_actor_loss(params, actions, obs, old_log_prob, advantage_func, eps=0.05):
        policy = actor_network.apply(params, obs)
        new_log_prob = policy.log_prob(actions)
        ratio = jnp.exp(new_log_prob - old_log_prob)
        clipped_ratio = jnp.clip(ratio, 1.0 - eps, 1.0 + eps)
        surr1 = clipped_ratio * advantage_func
        surr2 = ratio * advantage_func
        loss = jnp.minimum(surr1, surr2).mean()
        return -loss
    
    @jax.jit
    def actor_train_step(actor_params, actions, optim_state, obs, adv_function, old_log_prob, eps=0.05):
        loss, grads = jax.value_and_grad(compute_actor_loss)(actor_params, actions, obs, old_log_prob, adv_function, eps)
        updates, new_opt_state = actor_optim.update(grads, optim_state, actor_params)
        new_params = optax.apply_updates(actor_params, updates)
        return new_params, new_opt_state, loss

    @jax.jit
    def critic_train_step(critic_params, optim_state, obs, gt_rew_to_go):
        loss, grads = jax.value_and_grad(compute_critic_loss)(critic_params, obs, gt_rew_to_go)
        updates, new_opt_state = critic_optim.update(grads, optim_state, critic_params)
        new_params = optax.apply_updates(critic_params, updates)
        return new_params, new_opt_state, loss

    total_samples = num_envs * n_timesteps
    batch_size = total_samples // n_mini_batches

    for epoch in range(num_epochs):

        start_epoch_t = time.time()
        obs_buffer, actions_buff, log_prob_buff = [], [], []
        value_buff, reward_buff = [], []
        
        reset_batch(mjw_model, mjw_data, epoch_rngs[epoch])
        goal_cube_pos = find_goal_cube_pos(mj_model, mjw_data)
        obs = render_batch(mjw_model, mjw_data, render_ctx, rgb_buff)
        
        main_rng_key, subkey = jax.random.split(main_rng_key)
        timesteps_rng = jax.random.split(subkey, num=n_timesteps)

        for i in range(n_timesteps):
            policy = get_policy(actor_params, obs)
            value = get_value_estimates(critic_params, obs).squeeze(-1)
            action, log_prob = policy.sample_and_log_prob(seed=timesteps_rng[i])
            
            rew = step_batch(mj_model, mjw_model, mjw_data, action, goal_cube_pos)
            next_obs = render_batch(mjw_model, mjw_data, render_ctx, rgb_buff)
            
            reward_buff.append(rew)
            log_prob_buff.append(log_prob)
            obs_buffer.append(obs)
            actions_buff.append(action)
            value_buff.append(value)
            
            obs = next_obs

        delta_t = time.time() - start_epoch_t
        SPS = int(total_samples / delta_t)

        obs_arr = jnp.stack(obs_buffer, axis=1)
        actions_arr = jnp.stack(actions_buff, axis=1)
        log_prob_arr = jnp.stack(log_prob_buff, axis=1)
        value_arr = jnp.stack(value_buff, axis=1)
        reward_arr = jnp.stack(reward_buff, axis=1)

        rew_to_go = compute_rew_to_go(reward_arr, gamma_)

        mean_episode_rew = jnp.mean(rew_to_go)
        
        advantage_estimates = compute_advantage_estimates(value_arr, reward_arr, gamma_, lambda_)

        obs_flat = obs_arr.reshape((total_samples, *image_res, 3))
        actions_flat = actions_arr.reshape((total_samples, -1))
        log_prob_flat = log_prob_arr.reshape((total_samples,))
        rew_to_go_flat = rew_to_go.reshape((total_samples,))
        adv_flat = advantage_estimates.reshape((total_samples,))

        adv_flat = (adv_flat - jnp.mean(adv_flat)) / (jnp.std(adv_flat) + 1e-8)

        epoch_actor_loss = 0.0
        epoch_critic_loss = 0.0

        for n in range(n_mini_batches):
            start_idx = n * batch_size
            end_idx = (n + 1) * batch_size

            batch_obs = obs_flat[start_idx:end_idx]         
            batch_act = actions_flat[start_idx:end_idx]       
            batch_log_prob = log_prob_flat[start_idx:end_idx] 
            batch_rew_to_go = rew_to_go_flat[start_idx:end_idx]
            batch_adv = adv_flat[start_idx:end_idx]           

            actor_params, actor_opt_state, actor_loss = actor_train_step(
                actor_params, 
                batch_act, 
                actor_opt_state,
                batch_obs,
                batch_adv,
                batch_log_prob,
                eps
            )
            critic_params, critic_opt_state, critic_loss = critic_train_step(
                critic_params,
                critic_opt_state,
                batch_obs,
                batch_rew_to_go
            )
            epoch_actor_loss += actor_loss
            epoch_critic_loss += critic_loss

        mean_actor_loss = epoch_actor_loss / n_mini_batches
        mean_critic_loss = epoch_critic_loss / n_mini_batches

        print(f"Epoch {epoch + 1}/{num_epochs} | mean ep rew: {mean_episode_rew} | SPS: {SPS} | Actor Loss: {mean_actor_loss:.4f} | Critic Loss: {mean_critic_loss:.4f}")

    
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--gamma_", type=float, default=0.99)
    parser.add_argument("--num_epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_freq", type=int, default=10)
    parser.add_argument("--image_res", type=tuple, default=(128, 128))
    parser.add_argument("--n_timesteps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_mini_batches", type=int, default=16)

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
            args.seed,
            args.n_mini_batches
        )
    else:
        print("No CUDA-capable device available.")