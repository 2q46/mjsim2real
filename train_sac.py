import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import warp as wp
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp
import wandb
import mediapy as media

from env.mjenv import (
    find_goal_cube_pos,
    get_cube_id,
    get_gripper_id,
    init_mujoco,
    init_rendering,
    render_batch,
    reset_batch,
    step_batch,
    get_touch_sensor_adr,
)
from rl.sac_rgb import (
    SACActorConfig,
    SACActorNetwork,
    SACCriticConfig,
    SACCriticNetwork,
    compute_target_q,
    critic_train_step,
    actor_train_step,
    alpha_train_step,
    soft_update,
)


# --------------------------------------------------------------------------
# Replay buffer (host-side numpy, sampled + moved to device per update)
# --------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity, image_res, num_actuators):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        obs_shape = (capacity, *image_res, 3)
        self.obs = np.empty(obs_shape, dtype=np.uint8)
        self.next_obs = np.empty(obs_shape, dtype=np.uint8)
        self.action = np.empty((capacity, num_actuators), dtype=np.float32)
        self.reward = np.empty((capacity,), dtype=np.float32)
        self.done = np.empty((capacity,), dtype=np.float32)

    def add_batch(self, obs, action, reward, next_obs, done):
        n = obs.shape[0]
        idx = (self.ptr + np.arange(n)) % self.capacity

        self.obs[idx] = np.asarray(obs, dtype=np.uint8)
        self.next_obs[idx] = np.asarray(next_obs, dtype=np.uint8)
        self.action[idx] = np.asarray(action, dtype=np.float32)
        self.reward[idx] = np.asarray(reward, dtype=np.float32)
        self.done[idx] = np.asarray(done, dtype=np.float32)

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.action[idx],
            self.reward[idx],
            self.next_obs[idx],
            self.done[idx],
        )


# --------------------------------------------------------------------------
# Twin-critic helpers: pack two independent SACCriticNetwork param sets into
# one pytree and expose a single apply_fn returning (q1, q2), so the
# jitted step functions in rl/sac.py can stay critic-count agnostic.
# --------------------------------------------------------------------------

def make_twin_critic_apply_fn(critic_network):
    def apply_fn(params, obs, action):
        q1 = critic_network.apply(params["q1"], obs, action)
        q2 = critic_network.apply(params["q2"], obs, action)
        return q1, q2
    return apply_fn


def _save_video_and_checkpoints(epoch, obs_arr, actor_state, critic_params, checkpoint_dir):
    Path("videos").mkdir(parents=True, exist_ok=True)
    for index in range(int(obs_arr.shape[0])):
        media.write_video(f"videos/epoch_{epoch}_{index}.mp4", obs_arr[index], fps=24)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(checkpoint_dir / f"actor_{epoch}", actor_state)
    checkpointer.save(checkpoint_dir / f"critic_{epoch}", critic_params)
    checkpointer.wait_until_finished()


def main(
    num_envs: int = 32,
    image_res: list = [128, 128],
    num_actuators: int = 6,
    n_timesteps: int = 300,
    num_epochs: int = 1500,
    seed: int = 42,
    lr: float = 3e-4,
    alpha_lr: float = 3e-4,
    gamma_: float = 0.99,
    tau: float = 0.005,
    buffer_capacity: int = 100_000,
    batch_size: int = 256,
    updates_per_step: int = 1,
    warmup_steps: int = 5_000,
    checkpoint_freq: int = 10,
    target_entropy_scale: float = 1.0,
    init_log_alpha: float = 0.0,
):
    wandb.init(
        project="PickCube-mjsim2real-rl",
        tags=["sac"],
        config={
            "epochs": num_epochs,
            "gamma": gamma_,
            "tau": tau,
            "train_envs": num_envs,
            "image_res": image_res,
            "lr": lr,
            "alpha_lr": alpha_lr,
            "buffer_capacity": buffer_capacity,
            "batch_size": batch_size,
            "updates_per_step": updates_per_step,
            "warmup_steps": warmup_steps,
        },
    )

    image_res_tuple = tuple(image_res)

    mj_model, mjw_model, mjw_data = init_mujoco(num_envs)
    render_ctx, rgb_buff = init_rendering(mj_model, num_envs, image_res_tuple)

    main_rng_key = jax.random.PRNGKey(seed)
    main_rng_key, actor_rng, critic1_rng, critic2_rng = jax.random.split(main_rng_key, 4)
    np_rng = np.random.default_rng(seed)

    actor_cfg = SACActorConfig(img_size=image_res_tuple[0], output_features=num_actuators)
    critic_cfg = SACCriticConfig(img_size=image_res_tuple[0])

    actor_network = SACActorNetwork(cfg=actor_cfg)
    critic_network = SACCriticNetwork(cfg=critic_cfg)
    critic_apply_fn = make_twin_critic_apply_fn(critic_network)

    dummy_obs = jnp.zeros((1, *image_res_tuple, 3), dtype=jnp.float32)
    dummy_act = jnp.zeros((1, num_actuators), dtype=jnp.float32)

    actor_params = actor_network.init(actor_rng, dummy_obs)
    critic_params = {
        "q1": critic_network.init(critic1_rng, dummy_obs, dummy_act),
        "q2": critic_network.init(critic2_rng, dummy_obs, dummy_act),
    }
    critic_target_params = jax.tree_util.tree_map(lambda x: x, critic_params)

    actor_optim = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(learning_rate=lr, eps=1e-5))
    critic_optim = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(learning_rate=lr, eps=1e-5))
    alpha_optim = optax.adam(learning_rate=alpha_lr)

    actor_opt_state = actor_optim.init(actor_params)
    critic_opt_state = critic_optim.init(critic_params)

    log_alpha = jnp.asarray(init_log_alpha, dtype=jnp.float32)
    alpha_opt_state = alpha_optim.init(log_alpha)
    target_entropy = -target_entropy_scale * num_actuators

    eval_policy = jax.jit(actor_network.apply)

    replay_buffer = ReplayBuffer(buffer_capacity, image_res_tuple, num_actuators)

    cube_id = get_cube_id(mj_model)
    gripper_id = get_gripper_id(mj_model)
    fixed_touch_addr = get_touch_sensor_adr(mj_model, "fixed_jaw_touch")
    moving_touch_addr = get_touch_sensor_adr(mj_model, "moving_jaw_touch")

    global_step = 0
    actor_loss, critic_loss, alpha_loss = 0.0, 0.0, 0.0

    for i in range(num_epochs):
        print("=" * 50)
        epoch_num = i + 1
        print(f"Epoch number {epoch_num}")
        start_t = time.time()

        timestep_keys = jax.random.split(main_rng_key, num=n_timesteps + 4)
        reset_key, main_rng_key = timestep_keys[-1], timestep_keys[-2]
        sample_keys = timestep_keys[:n_timesteps]

        reset_batch(mj_model, mjw_model, mjw_data, reset_key, cube_id)
        goal_cube_pos = find_goal_cube_pos(mj_model, mjw_data)
        obs = render_batch(mjw_model, mjw_data, render_ctx, rgb_buff)

        epoch_rewards = []
        is_checkpoint_epoch = (i % checkpoint_freq == 0) and i >= 0
        video_frames = [] if is_checkpoint_epoch else None
        num_video_envs = min(4, num_envs)

        for t in range(n_timesteps):
            prev_ctrl = wp.clone(mjw_data.ctrl)

            if global_step < warmup_steps:
                action = jax.random.uniform(sample_keys[t], (num_envs, num_actuators), minval=-1.0, maxval=1.0)
            else:
                policy = eval_policy(actor_params, obs)
                action, _ = policy.sample_and_log_prob(seed=sample_keys[t])

            rew, is_touching, is_grasped, is_success = step_batch(
                cube_id, gripper_id, mjw_model, mjw_data, action, goal_cube_pos,
                prev_ctrl, fixed_touch_addr, moving_touch_addr,
            )
            new_obs = render_batch(mjw_model, mjw_data, render_ctx, rgb_buff)

            # NOTE: this environment resets once per epoch rather than per
            # episode, so there's no natural per-step terminal signal here.
            # `done` is left at 0 throughout; wire in your own termination
            # flag (e.g. from is_success) if the task should bootstrap
            # differently on success/failure.
            done = jnp.zeros((num_envs,), dtype=jnp.float32)

            replay_buffer.add_batch(obs, action, rew, new_obs, done)
            epoch_rewards.append(np.asarray(rew))

            if is_checkpoint_epoch:
                video_frames.append(np.asarray(new_obs[:num_video_envs], dtype=np.uint8))

            obs = new_obs
            global_step += num_envs

            if global_step >= warmup_steps:
                for _ in range(updates_per_step):
                    batch_obs, batch_act, batch_rew, batch_next_obs, batch_done = replay_buffer.sample(
                        batch_size, np_rng
                    )
                    batch_obs = jnp.asarray(batch_obs)
                    batch_act = jnp.asarray(batch_act)
                    batch_rew = jnp.asarray(batch_rew)
                    batch_next_obs = jnp.asarray(batch_next_obs)
                    batch_done = jnp.asarray(batch_done)

                    main_rng_key, target_key, actor_key = jax.random.split(main_rng_key, 3)

                    target_q = compute_target_q(
                        actor_params,
                        actor_network.apply,
                        critic_apply_fn,
                        critic_target_params,
                        log_alpha,
                        batch_next_obs,
                        batch_rew,
                        batch_done,
                        gamma_,
                        target_key,
                    )

                    critic_params, critic_opt_state, critic_loss, q1, q2 = critic_train_step(
                        critic_params,
                        critic_apply_fn,
                        critic_optim.update,
                        critic_opt_state,
                        batch_obs,
                        batch_act,
                        target_q,
                    )

                    actor_params, actor_opt_state, actor_loss, log_prob = actor_train_step(
                        actor_params,
                        actor_network.apply,
                        critic_apply_fn,
                        critic_params,
                        actor_optim.update,
                        actor_opt_state,
                        log_alpha,
                        batch_obs,
                        actor_key,
                        target_entropy,
                    )

                    log_alpha, alpha_opt_state, alpha_loss = alpha_train_step(
                        log_alpha, alpha_optim.update, alpha_opt_state, log_prob, target_entropy
                    )

                    critic_target_params = soft_update(critic_target_params, critic_params, tau)

        SPS = (n_timesteps * num_envs) / (time.time() - start_t)
        mean_episode_rew = float(np.mean(np.sum(np.stack(epoch_rewards, axis=1), axis=1)))

        if is_checkpoint_epoch and video_frames:
            # video_frames is a list of length n_timesteps, each (num_video_envs, H, W, 3).
            # Stack into (num_video_envs, n_timesteps, H, W, 3) so obs_arr[index] is a
            # proper (T, H, W, 3) clip for media.write_video.
            obs_arr = np.stack(video_frames, axis=1)

            actor_state = TrainState(
                step=i, apply_fn=actor_network.apply, params=actor_params, tx=actor_optim, opt_state=actor_opt_state
            )
            checkpoint_dir = Path("checkpoints/").absolute()
            _save_video_and_checkpoints(i, obs_arr, actor_state, critic_params, checkpoint_dir)

        alpha_value = float(jnp.exp(log_alpha))

        wandb.log({
            "train/actor_loss": float(actor_loss),
            "train/critic_loss": float(critic_loss),
            "train/alpha_loss": float(alpha_loss),
            "train/alpha": alpha_value,
            "train/SPS": SPS,
            "train/mean_episode_rew": mean_episode_rew,
            "train/global_step": global_step,
            "train/buffer_size": replay_buffer.size,
        })

        print(f"mean rew: {mean_episode_rew:.3f}")
        print(f"alpha: {alpha_value:.4f}")
        print(f"buffer size: {replay_buffer.size}")
        print(f"steps per second: {SPS:.1f}")
        print("=" * 50)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=196)
    parser.add_argument("--image_res", nargs=2, type=int, default=[128, 128])
    parser.add_argument("--num_actuators", type=int, default=6)
    parser.add_argument("--n_timesteps", type=int, default=300)
    parser.add_argument("--num_epochs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=402)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--gamma_", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--buffer_capacity", type=int, default=150_000)
    parser.add_argument("--batch_size", type=int, default=3_500)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--checkpoint_freq", type=int, default=10)

    args = parser.parse_args()

    if len(jax.devices("cuda")) > 0:
        print("CUDA-capable device available...")
        main(
            num_envs=args.num_envs,
            image_res=args.image_res,
            num_actuators=args.num_actuators,
            n_timesteps=args.n_timesteps,
            num_epochs=args.num_epochs,
            seed=args.seed,
            lr=args.lr,
            alpha_lr=args.alpha_lr,
            gamma_=args.gamma_,
            tau=args.tau,
            buffer_capacity=args.buffer_capacity,
            batch_size=args.batch_size,
            updates_per_step=args.updates_per_step,
            warmup_steps=args.warmup_steps,
            checkpoint_freq=args.checkpoint_freq,
        )
    else:
        print("No CUDA-capable device available.")