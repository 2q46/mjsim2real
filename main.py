import os

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

import argparse
from pathlib import Path
from functools import partial
import time
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp
import wandb
import mediapy as media
import numpy as np

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
    compute_adv_estimates,
    compute_advantage_estimates,
    compute_rew_to_go,
    compute_eval_metrics
)

def compute_actor_loss(params, apply_fn, actions, obs_float, old_log_prob, advantage_func, eps=0.05):
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
    total_loss = policy_loss
    return total_loss


def compute_critic_loss(params, apply_fn, obs_float, gt_rew_to_go):
    estimated_rew_to_go = apply_fn(params, obs_float)
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
        actor_params, actor_apply_fn, batch_act, batch_obs, batch_log_prob, batch_adv, eps
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


@jax.jit
def update_buffers(
    obs_buffer,
    act_buffer,
    rew_buffer,
    val_buffer,
    log_prob_buffer,
    is_success_buffer,
    is_touching_buffer,
    is_grasped_buffer,
    obs,
    act,
    rew,
    val,
    log_prob,
    is_success,
    is_touching,
    is_grasped,
    t
):
    
    obs_buffer = obs_buffer.at[:, t].set(obs.astype(jnp.uint8))
    act_buffer = act_buffer.at[:, t].set(act)
    rew_buffer = rew_buffer.at[:, t].set(rew)
    val_buffer = val_buffer.at[:, t].set(val)
    is_success_buffer = is_success_buffer.at[:, t].set(is_success)
    is_touching_buffer = is_touching_buffer.at[:, t].set(is_touching)
    is_grasped_buffer = is_grasped_buffer.at[:, t].set(is_grasped)

    log_prob_buffer = log_prob_buffer.at[:, t].set(log_prob)
    return (
        obs_buffer,
        act_buffer,
        rew_buffer,
        val_buffer,
        log_prob_buffer,
        is_success_buffer,
        is_touching_buffer,
        is_grasped_buffer
    )


@partial(jax.jit, static_argnums=[7, 8])
def flatten_buffers(
    obs_buffer,
    act_buffer,
    rew_buffer,
    val_buffer,
    log_prob_buffer,
    adv_estimates,
    rew_to_go,
    image_res,
    total_samples
):
    rew_to_go = jnp.reshape(rew_to_go, (total_samples))
    adv_estimates = jnp.reshape(adv_estimates, (total_samples))
    val_buffer = jnp.reshape(val_buffer, (total_samples))
    obs_buffer = jnp.reshape(obs_buffer, (total_samples, *image_res, 3))
    act_buffer = jnp.reshape(act_buffer, (total_samples, 6))
    rew_buffer = jnp.reshape(rew_buffer, (total_samples))
    log_prob_buffer = jnp.reshape(log_prob_buffer, (total_samples))
    return (
        obs_buffer,
        act_buffer,
        rew_buffer,
        val_buffer,
        log_prob_buffer,
        adv_estimates,
        rew_to_go
    )

def main(args):

    wandb.init(
        project="PickCube-mjsim2real-rl",
        tags=["ppo"],
        config={
            "epochs": args.num_epochs, 
            "gamma": args.gamma_,
            "lambda": args.lambda_,
            "train_envs": args.num_envs,
            "image_res": args.image_res,
            "lr": args.lr,
            "eps": args.eps
        }
    )
    checkpointer = ocp.StandardCheckpointer()

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

    eval_policy = jax.jit(actor_network.apply)
    eval_value = jax.jit(critic_network.apply)
    
    cube_id = get_cube_id(mj_model)
    gripper_id = get_gripper_id(mj_model)
    
    for i in range(args.num_epochs):

        print("="*50)
        epoch_num = i+1
        print(f"Epoch number {epoch_num}")
        start_t = time.time()

        obs_buffer = jnp.empty((args.num_envs, args.n_timesteps, *image_res, 3), dtype=jnp.uint8)
        act_buffer = jnp.empty((args.num_envs, args.n_timesteps, 6), dtype=jnp.float32)
        log_prob_buffer = jnp.empty((args.num_envs, args.n_timesteps), dtype=jnp.float32)
        rew_buffer = jnp.empty((args.num_envs, args.n_timesteps), dtype=jnp.float32)
        val_buffer = jnp.empty((args.num_envs, args.n_timesteps), dtype=jnp.float32)
        success_buffer = jnp.empty((args.num_envs, args.n_timesteps), dtype=jnp.int8)
        is_touching_buffer = jnp.empty((args.num_envs, args.n_timesteps), dtype=jnp.int8)
        is_grasped_buffer = jnp.empty((args.num_envs, args.n_timesteps), dtype=jnp.int8)

        timestep_keys = jax.random.split(main_rng_key, num=args.n_timesteps)
        reset_key, main_rng_key = jax.random.split(main_rng_key, num=2)

        reset_batch(mjw_model, mjw_data, reset_key)
        goal_cube_pos = find_goal_cube_pos(mj_model, mjw_data)
        obs = render_batch(mjw_model, mjw_data, render_ctx, rgb_buff)

        for t in range(args.n_timesteps):

            policy = eval_policy(actor_params, obs)
            value = eval_value(critic_params, obs)
            actions, log_prob = policy.sample_and_log_prob(seed=timestep_keys[t])
            rew, is_touching, is_grasped, is_success = step_batch(cube_id, gripper_id, mjw_model, mjw_data, actions, goal_cube_pos)
            new_obs = render_batch(mjw_model, mjw_data, render_ctx, rgb_buff)
            obs_buffer, act_buffer, rew_buffer, val_buffer, log_prob_buffer, success_buffer, is_touching_buffer, is_grasped_buffer = update_buffers(
                    obs_buffer,
                    act_buffer,
                    rew_buffer,
                    val_buffer,
                    log_prob_buffer,
                    success_buffer,
                    is_touching_buffer,
                    is_grasped_buffer,
                    obs,
                    actions,
                    rew,
                    value,
                    log_prob,
                    is_success,
                    is_touching,
                    is_grasped,
                    t
            )
            obs = new_obs

        SPS = (args.n_timesteps * args.num_envs) / (time.time() - start_t)

        if i % args.checkpoint_freq == 0 and i >= 1150: 
        
            obs_arr = np.asarray(obs_buffer[0:16], dtype=np.uint8)
            for index in range(int(obs_arr.shape[0])):
                video_frames = np.ascontiguousarray(obs_arr[index])
                media.write_video(f"videos/epoch_{i}_{index}.mp4", video_frames, fps=24)
            checkpoint_dir = Path("checkpoints/").absolute()
        
            actor_state = TrainState(
                step=i,
                apply_fn=actor_network.apply,
                params=actor_params,
                tx=actor_optim,
                opt_state=actor_opt_state
            )
            critic_state = TrainState(
                step=i,
                apply_fn=critic_network.apply,
                params=critic_params,
                tx=critic_optim,
                opt_state=critic_opt_state
            )
            checkpointer.save(checkpoint_dir / f"actor_{i}", actor_state)
            checkpointer.wait_until_finished()
            checkpointer.save(checkpoint_dir / f"critic_{i}", critic_state)
            checkpointer.wait_until_finished()

        rew_to_go, mean_episode_rew = compute_rew_to_go(rew_buffer, args.gamma_)
        adv_estimates = compute_advantage_estimates(val_buffer, rew_buffer, args.gamma_, args.lambda_)
        num_success, num_is_touching, num_is_grasped = compute_eval_metrics(success_buffer, is_touching_buffer, is_grasped_buffer)
        obs_buffer, act_buffer, rew_buffer, val_buffer, log_prob_buffer, adv_estimates, rew_to_go = flatten_buffers(
            obs_buffer, 
            act_buffer, 
            rew_buffer, 
            val_buffer, 
            log_prob_buffer, 
            adv_estimates, 
            rew_to_go, 
            image_res, 
            total_samples
        )

        mean_actor_loss, mean_critic_loss = 0.0, 0.0
        
        for n in range(args.n_mini_batches):

            max_idx, min_idx = ((n+1) * batch_size), (n * batch_size)

            sampled_obs_buff = obs_buffer[min_idx: max_idx]
            sampled_batch_act = act_buffer[min_idx: max_idx]
            sampled_batch_log_prob = log_prob_buffer[min_idx: max_idx]
            sampled_batch_adv = adv_estimates[min_idx: max_idx]
            sampled_batch_rew_to_go = rew_to_go[min_idx: max_idx]

            actor_params, critic_params, actor_opt_state, critic_opt_state, actor_loss, critic_loss = actor_critic_train_step(
                actor_params,
                critic_params,
                actor_network.apply,
                critic_network.apply,
                actor_optim.update,
                critic_optim.update,
                actor_opt_state,
                critic_opt_state,
                sampled_obs_buff,
                sampled_batch_act,
                sampled_batch_log_prob,
                sampled_batch_adv,
                sampled_batch_rew_to_go,
                args.eps,
                args.ent_coef,
            )
            mean_actor_loss += actor_loss
            mean_critic_loss += critic_loss

        mean_actor_loss /= args.n_mini_batches
        mean_critic_loss /= args.n_mini_batches

        wandb.log({
            "train/actor_loss": mean_actor_loss,
            "train/critic_loss": mean_critic_loss,
            "train/SPS": SPS,
            "train/mean_episode_rew": mean_episode_rew,
            "eval/num_success": num_success,
            "eval/num_touching": num_is_touching,
            "eval/num_grasped": num_is_grasped,
        })

        print(f"actor loss: {mean_actor_loss}")
        print(f"critic loss: {mean_critic_loss}")
        print(f"mean rew: {mean_episode_rew}")
        print(f"num is touching: {num_is_touching}")
        print(f"num success: {num_success}")
        print(f"num grasped: {num_is_grasped}")
        print(f"steps per second: {SPS}")

        print("="*50)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--gamma_", type=float, default=0.99)
    parser.add_argument("--num_epochs", type=int, default=1500)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_freq", type=int, default=10)
    parser.add_argument("--image_res", nargs=2, type=int, default=[128, 128])
    parser.add_argument("--n_timesteps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--n_mini_batches", type=int, default=8)  

    args = parser.parse_args()

    if len(jax.devices("cuda")) > 0:
        print("CUDA-capable device available...")
        main(args)
    else:
        print("No CUDA-capable device available.")