import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"


import argparse
from functools import partial
import time
import jax
import jax.numpy as jnp
import optax

from env.mjenv import (
    find_goal_cube_pos,
    get_cube_id,
    get_gripper_id,
    init_mujoco,
    init_rendering,
    render_batch,
    reset_batch,
    step_batch,
)
from rl.ppo_rgb import (
    ActorConfig,
    ActorNetwork,
    CriticConfig,
    CriticNetwork,
    compute_advantage_estimates,
    compute_rew_to_go,
)


def compute_actor_loss(params, apply_fn, actions, obs_float, old_log_prob, advantage_func, eps=0.05, ent_coef=0.01):
    policy = apply_fn(params, obs_float)
    new_log_prob = policy.log_prob(actions)
    old_log_prob = jnp.reshape(old_log_prob, new_log_prob.shape)
    advantage_func = jnp.reshape(advantage_func, new_log_prob.shape)
    log_ratio = new_log_prob - old_log_prob
    ratio = jnp.exp(log_ratio)
    clipped_ratio = jnp.clip(ratio, 1.0 - eps, 1.0 + eps)
    surr1 = ratio * advantage_func
    surr2 = clipped_ratio * advantage_func
    policy_loss = -jnp.minimum(surr1, surr2).mean()
    entropy_loss = policy.entropy().mean()
    total_loss = policy_loss - (ent_coef * entropy_loss)
    return total_loss


def compute_critic_loss(params, apply_fn, obs_float, gt_rew_to_go):
    estimated_rew_to_go = apply_fn(params, obs_float).squeeze(-1)
    gt_rew_to_go = jnp.reshape(gt_rew_to_go, estimated_rew_to_go.shape)
    return jnp.mean(jnp.square(gt_rew_to_go - estimated_rew_to_go))

@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def actor_critic_train_step(
    actor_params,
    critic_params,
    actor_apply_fn,
    critic_apply_fn,
    actor_tx_update,
    critic_tx_update,
    actor_opt_state,
    critic_opt_state,
    batch_obs,
    batch_act,
    batch_log_prob,
    batch_adv,
    batch_rew_to_go,
    eps,
    ent_coef,
):
    actor_loss, actor_grads = jax.value_and_grad(compute_actor_loss)(
        actor_params, actor_apply_fn, batch_act, batch_obs, batch_log_prob, batch_adv, eps, ent_coef
    )
    actor_updates, new_actor_opt_state = actor_tx_update(actor_grads, actor_opt_state, actor_params)
    new_actor_params = optax.apply_updates(actor_params, actor_updates)

    critic_loss, critic_grads = jax.value_and_grad(compute_critic_loss)(
        critic_params, critic_apply_fn, batch_obs, batch_rew_to_go
    )
    critic_updates, new_critic_opt_state = critic_tx_update(critic_grads, critic_opt_state, critic_params)
    new_critic_params = optax.apply_updates(critic_params, critic_updates)

    return (
        new_actor_params,
        new_critic_params,
        new_actor_opt_state,
        new_critic_opt_state,
        actor_loss,
        critic_loss,
    )


def main(args):
    num_envs = args.num_envs
    image_res = tuple(args.image_res)

    mj_model, mjw_model, mjw_data = init_mujoco(num_envs)
    render_ctx, rgb_buff = init_rendering(mj_model, num_envs, image_res)

    main_rng_key = jax.random.PRNGKey(args.seed)
    main_rng_key, actor_rng, critic_rng = jax.random.split(main_rng_key, 3)

    actor_cfg = ActorConfig(img_size=image_res)
    critic_cfg = CriticConfig(img_size=image_res)

    actor_network = ActorNetwork(cfg=actor_cfg)
    critic_network = CriticNetwork(cfg=critic_cfg)

    dummy_obs = jnp.zeros((1, *image_res, 3), dtype=jnp.float32)
    actor_params = actor_network.init(actor_rng, dummy_obs)
    critic_params = critic_network.init(critic_rng, dummy_obs)

    actor_optim = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=args.lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=args.lr, eps=1e-5),
    )

    actor_opt_state = actor_optim.init(actor_params)
    critic_opt_state = critic_optim.init(critic_params)

    total_samples = num_envs * args.n_timesteps
    batch_size = total_samples // args.n_mini_batches

    cube_id = get_cube_id(mj_model)
    gripper_id = get_gripper_id(mj_model)

    eval_policy = jax.jit(actor_network.apply)
    eval_value = jax.jit(critic_network.apply)

    for epoch in range(args.num_epochs):
        start_epoch_t = time.time()

        main_rng_key, reset_key, rollout_key = jax.random.split(main_rng_key, 3)
        reset_batch(mjw_model, mjw_data, reset_key)
        goal_cube_pos = find_goal_cube_pos(mj_model, mjw_data)

        obs_buffer, actions_buff, log_prob_buff = [], [], []
        value_buff, reward_buff = [], []

        obs = render_batch(mjw_model, mjw_data, render_ctx, rgb_buff)
        timesteps_rng = jax.random.split(rollout_key, num=args.n_timesteps)

        for i in range(args.n_timesteps):
            # Convert to float32 [0, 1] during inference pass
            obs_float = obs.astype(jnp.float32) / 255.0

            policy = eval_policy(actor_params, obs_float)
            value = eval_value(critic_params, obs_float).squeeze(-1)

            action, log_prob = policy.sample_and_log_prob(seed=timesteps_rng[i])

            rew = step_batch(cube_id, gripper_id, mjw_model, mjw_data, action, goal_cube_pos)
            next_obs = render_batch(mjw_model, mjw_data, render_ctx, rgb_buff)

            reward_buff.append(rew)
            log_prob_buff.append(log_prob)
            obs_buffer.append(obs.astype(jnp.uint8))
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

        rew_to_go = compute_rew_to_go(reward_arr, args.gamma_)
        advantage_estimates = compute_advantage_estimates(value_arr, reward_arr, args.gamma_, args.lambda_)

        obs_flat = obs_arr.reshape((total_samples, *image_res, 3))
        actions_flat = actions_arr.reshape((total_samples, -1))
        log_prob_flat = log_prob_arr.reshape((total_samples,))
        rew_to_go_flat = rew_to_go.reshape((total_samples,))
        adv_flat = advantage_estimates.reshape((total_samples,))

        adv_flat = (adv_flat - jnp.mean(adv_flat)) / (jnp.std(adv_flat) + 1e-8)

        epoch_actor_loss = 0.0
        epoch_critic_loss = 0.0

        for _ in range(args.ppo_epochs):
            main_rng_key, perm_key = jax.random.split(main_rng_key)
            perm_indices = jax.random.permutation(perm_key, total_samples)

            for n in range(args.n_mini_batches):
                start_idx = n * batch_size
                end_idx = (n + 1) * batch_size
                batch_inds = perm_indices[start_idx:end_idx]

                batch_obs = obs_flat[batch_inds]  # Keep in uint8 until inside train step
                batch_act = actions_flat[batch_inds]
                batch_log_prob = log_prob_flat[batch_inds]
                batch_rew_to_go = rew_to_go_flat[batch_inds]
                batch_adv = adv_flat[batch_inds]

                (
                    actor_params,
                    critic_params,
                    actor_opt_state,
                    critic_opt_state,
                    actor_loss,
                    critic_loss,
                ) = actor_critic_train_step(
                    actor_params,
                    critic_params,
                    actor_network.apply,
                    critic_network.apply,
                    actor_optim.update,
                    critic_optim.update,
                    actor_opt_state,
                    critic_opt_state,
                    batch_obs,
                    batch_act,
                    batch_log_prob,
                    batch_adv,
                    batch_rew_to_go,
                    args.eps,
                    args.ent_coef,
                )

                # 3. Block async execution queue from piling up in VRAM
                actor_loss.block_until_ready()

                epoch_actor_loss += actor_loss
                epoch_critic_loss += critic_loss

        total_updates = args.ppo_epochs * args.n_mini_batches
        mean_actor_loss = epoch_actor_loss / total_updates
        mean_critic_loss = epoch_critic_loss / total_updates
        mean_ep_rew = jnp.mean(rew_to_go)

        print(
            f"Epoch {epoch + 1}/{args.num_epochs} | "
            f"Ep Rew: {mean_ep_rew:.2f} | SPS: {SPS} | "
            f"Actor Loss: {mean_actor_loss:.4f} | Critic Loss: {mean_critic_loss:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--gamma_", type=float, default=0.99)
    parser.add_argument("--num_epochs", type=int, default=400)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint_freq", type=int, default=10)
    parser.add_argument("--image_res", nargs=2, type=int, default=[128, 128])
    parser.add_argument("--n_timesteps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_mini_batches", type=int, default=32)  # Increased from 16 to 32

    args = parser.parse_args()

    if len(jax.devices("cuda")) > 0:
        print("CUDA-capable device available...")
        main(args)
    else:
        print("No CUDA-capable device available.")