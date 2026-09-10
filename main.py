import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import warp as wp
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
from distrax import MultivariateNormalDiag

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
    compute_eval_metrics
)

def compute_actor_loss(params, apply_fn, actions, obs_float, old_log_prob, advantage_func, eps=0.1, ent_coef=0.01):
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
    entropy = policy.entropy().mean()
    
    total_loss = policy_loss - (ent_coef * entropy)
    return total_loss

def compute_critic_loss(params, apply_fn, obs_float, gt_rew_to_go):
    estimated_rew_to_go = apply_fn(params, obs_float)
    
    target_mean = jnp.mean(gt_rew_to_go)
    target_std = jnp.std(gt_rew_to_go) + 1e-8
    norm_gt = (gt_rew_to_go - target_mean) / target_std
    norm_pred = (estimated_rew_to_go - target_mean) / target_std
    
    loss = optax.huber_loss(norm_pred, norm_gt).mean()
    return loss

@jax.jit
def shuffle_buffers(
    obs_buffer,
    act_buffer,
    log_prob_buffer,
    adv_estimates,
    rew_to_go,
    main_key
):
    shuffle_key, main_key = jax.random.split(main_key)
    perm = jax.random.permutation(shuffle_key, obs_buffer.shape[0])
    
    obs_buffer = obs_buffer[perm]
    act_buffer = act_buffer[perm]
    rew_to_go = rew_to_go[perm]
    adv_estimates = adv_estimates[perm]
    log_prob_buffer = log_prob_buffer[perm]
    return (
        obs_buffer,
        act_buffer,
        rew_to_go,
        adv_estimates,
        log_prob_buffer,
        main_key
    )

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
    rew_to_go = jnp.reshape(rew_to_go, (total_samples,))
    adv_estimates = jnp.reshape(adv_estimates, (total_samples,))
    val_buffer = jnp.reshape(val_buffer, (total_samples,))
    obs_buffer = jnp.reshape(obs_buffer, (total_samples, *image_res, 3))
    act_buffer = jnp.reshape(act_buffer, (total_samples, 6))
    rew_buffer = jnp.reshape(rew_buffer, (total_samples,))
    log_prob_buffer = jnp.reshape(log_prob_buffer, (total_samples,))
    return (
        obs_buffer,
        act_buffer,
        rew_buffer,
        val_buffer,
        log_prob_buffer,
        adv_estimates,
        rew_to_go
    )

def _save_video_and_checkpoints(epoch, obs_arr, grasped_arr, success_arr, actor_state, critic_state, checkpoint_dir):
    Path("videos").mkdir(parents=True, exist_ok=True)
    for index in range(int(obs_arr.shape[0])):
        media.write_video(f"videos/epoch_{epoch}_{index}.mp4", obs_arr[index], fps=24)
        
    if grasped_arr is not None:
        for index in range(int(grasped_arr.shape[0])):
            media.write_video(f"videos/epoch_{epoch}_grasp_{index}.mp4", grasped_arr[index], fps=24)

    if success_arr is not None:
        for index in range(int(success_arr.shape[0])):
            media.write_video(f"videos/epoch_{epoch}_success_{index}.mp4", success_arr[index], fps=24)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(checkpoint_dir / f"actor_{epoch}", actor_state)
    checkpointer.save(checkpoint_dir / f"critic_{epoch}", critic_state)
    checkpointer.wait_until_finished()

def main(
    num_envs: int = 64,
    eps: float = 0.1,
    ent_coef: float = 0.01,
    lambda_: float = 0.95,
    gamma_: float = 0.95,
    num_epochs: int = 1500,
    ppo_epochs: int = 4,
    lr: float = 3e-4,
    checkpoint_freq: int = 10,
    image_res: list = [128, 128],
    n_timesteps: int = 300,
    seed: int = 42,
    n_mini_batches: int = 8,
    num_actuators: int = 6

):
    wandb.init(
        project="PickCube-mjsim2real-rl",
        tags=["ppo"],
        config={
            "epochs": num_epochs, 
            "ppo_epochs": ppo_epochs,
            "gamma": gamma_,
            "lambda": lambda_,
            "train_envs": num_envs,
            "image_res": image_res,
            "lr": lr,
            "eps": eps,
            "ent_coef": ent_coef
        }
    )

    image_res_tuple = tuple(image_res)

    mj_model, mjw_model, mjw_data = init_mujoco(num_envs)
    render_ctx, rgb_buff = init_rendering(mj_model, num_envs, image_res_tuple)

    main_rng_key = jax.random.PRNGKey(seed)
    main_rng_key, actor_rng, critic_rng = jax.random.split(main_rng_key, 3)

    actor_cfg = ActorConfig(img_size=image_res_tuple)
    critic_cfg = CriticConfig(img_size=image_res_tuple)

    actor_network = ActorNetwork(cfg=actor_cfg)
    critic_network = CriticNetwork(cfg=critic_cfg)

    dummy_obs = jnp.zeros((1, *image_res_tuple, 3), dtype=jnp.float32)
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
    total_samples = num_envs * n_timesteps
    batch_size = total_samples // n_mini_batches

    eval_policy = jax.jit(actor_network.apply)
    eval_value = jax.jit(critic_network.apply)
    
    cube_id = get_cube_id(mj_model)
    gripper_id = get_gripper_id(mj_model)
    
    for i in range(num_epochs):

        print("="*50)
        epoch_num = i + 1
        print(f"Epoch number {epoch_num}")
        start_t = time.time()

        obs_buffer = jnp.empty((num_envs, n_timesteps, *image_res_tuple, 3), dtype=jnp.uint8)
        act_buffer = jnp.empty((num_envs, n_timesteps, 6), dtype=jnp.float32)
        log_prob_buffer = jnp.empty((num_envs, n_timesteps), dtype=jnp.float32)
        rew_buffer = jnp.empty((num_envs, n_timesteps), dtype=jnp.float32)
        val_buffer = jnp.empty((num_envs, n_timesteps), dtype=jnp.float32)
        success_buffer = jnp.empty((num_envs, n_timesteps), dtype=jnp.uint8)
        is_touching_buffer = jnp.empty((num_envs, n_timesteps), dtype=jnp.uint8)
        is_grasped_buffer = jnp.empty((num_envs, n_timesteps), dtype=jnp.uint8)

        timestep_keys = jax.random.split(main_rng_key, num=n_timesteps)
        reset_key, main_rng_key = jax.random.split(main_rng_key, num=2)

        reset_batch(mj_model, mjw_model, mjw_data, reset_key, cube_id)
        goal_cube_pos = find_goal_cube_pos(mj_model, mjw_data)
        obs = render_batch(mjw_model, mjw_data, render_ctx, rgb_buff)

        for t in range(n_timesteps):

            prev_ctrl = wp.clone(mjw_data.ctrl)    
            policy = eval_policy(actor_params, obs)
            value = eval_value(critic_params, obs)
            actions, log_prob = policy.sample_and_log_prob(seed=timestep_keys[t])
            rew, is_touching, is_grasped, is_success = step_batch(cube_id, gripper_id, mjw_model, mjw_data, actions, goal_cube_pos, prev_ctrl)
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

        SPS = (n_timesteps * num_envs) / (time.time() - start_t)

        rew_to_go, mean_episode_rew = compute_rew_to_go(rew_buffer, gamma_)
        adv_estimates = compute_advantage_estimates(val_buffer, rew_buffer, gamma_, lambda_)
        num_success, num_is_touching, num_is_grasped = compute_eval_metrics(success_buffer, is_touching_buffer, is_grasped_buffer)

        if i % checkpoint_freq == 0 and i >= 0: 
            obs_arr = np.asarray(obs_buffer[0:4], dtype=np.uint8)
            grasped_non_zero = jnp.any(is_grasped_buffer, axis=1).nonzero()[0]
            success_non_zero = jnp.any(success_buffer, axis=1).nonzero()[0]

            grasped_arr = np.asarray(obs_buffer[grasped_non_zero[0:2]], dtype=np.uint8) if grasped_non_zero.shape[0] >= 2 else None
            success_arr = np.asarray(obs_buffer[success_non_zero[0:2]], dtype=np.uint8) if success_non_zero.shape[0] >= 2 else None

            actor_state = TrainState(
                step=i, apply_fn=actor_network.apply, params=actor_params, tx=actor_optim, opt_state=actor_opt_state
            )
            critic_state = TrainState(
                step=i, apply_fn=critic_network.apply, params=critic_params, tx=critic_optim, opt_state=critic_opt_state
            )
            checkpoint_dir = Path("checkpoints/").absolute()

            _save_video_and_checkpoints(
                i, obs_arr, grasped_arr, success_arr, actor_state, critic_state, checkpoint_dir
            )

        obs_buffer, act_buffer, rew_buffer, val_buffer, log_prob_buffer, adv_estimates, rew_to_go = flatten_buffers(
            obs_buffer, 
            act_buffer, 
            rew_buffer, 
            val_buffer, 
            log_prob_buffer, 
            adv_estimates, 
            rew_to_go, 
            image_res_tuple, 
            total_samples
        )

        mean_actor_loss, mean_critic_loss = 0.0, 0.0
        
        for ppo_epoch in range(ppo_epochs):
            (
                obs_buffer,
                act_buffer,
                rew_to_go,
                adv_estimates,
                log_prob_buffer,
                main_rng_key
            ) = shuffle_buffers(
                obs_buffer,
                act_buffer,
                log_prob_buffer,
                adv_estimates,
                rew_to_go,
                main_rng_key
            )

            for n in range(n_mini_batches):
                max_idx, min_idx = ((n + 1) * batch_size), (n * batch_size)

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
                    eps,
                    ent_coef,
                )
                mean_actor_loss += actor_loss
                mean_critic_loss += critic_loss

        total_updates = ppo_epochs * n_mini_batches
        mean_actor_loss /= total_updates
        mean_critic_loss /= total_updates

        wandb.log({
            "train/actor_loss": mean_actor_loss,
            "train/critic_loss": mean_critic_loss,
            "train/SPS": SPS,
            "train/mean_episode_rew": mean_episode_rew,
            "eval/num_success": num_success,
            "eval/num_touching": num_is_touching,
            "eval/num_grasped": num_is_grasped,
        })

        print(f"actor loss: {mean_actor_loss:.6f}")
        print(f"critic loss: {mean_critic_loss:.6f}")
        print(f"mean rew: {mean_episode_rew:.3f}")
        print(f"num is touching: {num_is_touching}")
        print(f"num success: {num_success}")
        print(f"num grasped: {num_is_grasped}")
        print(f"steps per second: {SPS:.1f}")

        print("=" * 50)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=96)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--ent_coef", type=float, default=0.005)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--gamma_", type=float, default=0.99)
    parser.add_argument("--num_epochs", type=int, default=1500)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_freq", type=int, default=10)
    parser.add_argument("--image_res", nargs=2, type=int, default=[128, 128])
    parser.add_argument("--n_timesteps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=402)
    parser.add_argument("--n_mini_batches", type=int, default=8)  

    args = parser.parse_args()

    if len(jax.devices("cuda")) > 0:
        print("CUDA-capable device available...")
        main(
            num_envs=args.num_envs,
            eps=args.eps,
            ent_coef=args.ent_coef,
            lambda_=args.lambda_,
            gamma_=args.gamma_,
            num_epochs=args.num_epochs,
            ppo_epochs=args.ppo_epochs,
            lr=args.lr,
            checkpoint_freq=args.checkpoint_freq,
            image_res=args.image_res,
            n_timesteps=args.n_timesteps,
            seed=args.seed,
            n_mini_batches=args.n_mini_batches,
        )
    else:
        print("No CUDA-capable device available.")