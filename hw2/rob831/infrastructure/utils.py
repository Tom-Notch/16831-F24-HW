import copy
import multiprocessing as mp
import time
from random import sample

import gym
import numpy as np
from rob831.infrastructure import pytorch_util as ptu

############################################
############################################


def calculate_mean_prediction_error(env, action_sequence, models, data_statistics):

    model = models[0]

    # true
    true_states = perform_actions(env, action_sequence)["observation"]

    # predicted
    ob = np.expand_dims(true_states[0], 0)
    pred_states = []
    for ac in action_sequence:
        pred_states.append(ob)
        action = np.expand_dims(ac, 0)
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)

    # mpe
    mpe = mean_squared_error(pred_states, true_states)

    return mpe, true_states, pred_states


def perform_actions(env, actions):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    for ac in actions:
        obs.append(ob)
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def mean_squared_error(a, b):
    return np.mean((a - b) ** 2)


############################################
############################################


def sample_trajectory(
    env, policy, max_path_length, render=False, render_mode=("rgb_array")
):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        if render:  # feel free to ignore this for now
            if "rgb_array" in render_mode:
                if hasattr(env.unwrapped, "sim"):
                    if "track" in env.unwrapped.model.camera_names:
                        image_obs.append(
                            env.unwrapped.sim.render(
                                camera_name="track", height=500, width=500
                            )[::-1]
                        )
                    else:
                        image_obs.append(
                            env.unwrapped.sim.render(height=500, width=500)[::-1]
                        )
                else:
                    image_obs.append(env.render(mode=render_mode))
            if "human" in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)

        # TODO: get this from hw1
        obs.append(ob)
        ac = policy.get_action(ob)
        ac = ac[0]
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done or steps > max_path_length:
            terminals.append(1)
            break
        else:
            terminals.append(0)
    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def sample_trajectories(
    env,
    policy,
    min_timesteps_per_batch,
    max_path_length,
    render=False,
    render_mode=("rgb_array"),
    parallel: bool = False,
    initial_num_workers: int = 10,
    max_num_workers: int = mp.cpu_count() - 1,
):
    # TODO: get this from hw1
    """
    Collect rollouts until we have collected min_timesteps_per_batch steps.
    """

    if parallel:
        # Serialize the policy parameters if necessary
        # For example, if using PyTorch or TensorFlow, extract model weights
        # We'll assume here that the policy can be deep-copied

        # Shared variables managed by multiprocessing.Manager()
        manager = mp.Manager()
        paths = manager.list()
        timesteps_counter = manager.Value("i", 0)
        lock = manager.Lock()

        print(f"Average path length: {sample_trajectories.average_path_length}")

        num_workers = (
            initial_num_workers
            if sample_trajectories.average_path_length is None
            else int(
                min(
                    min_timesteps_per_batch // sample_trajectories.average_path_length
                    + 1,
                    max_num_workers,
                )
            )
        )

        print(f"Using {num_workers} worker(s) for sampling trajectories.")

        # Create worker processes
        processes = []
        for worker_id in range(num_workers):
            p = mp.Process(
                target=worker_sample_trajectories,
                args=(
                    worker_id,
                    env.spec.id,
                    policy,
                    max_path_length,
                    render,
                    render_mode,
                    lock,
                    timesteps_counter,
                    paths,
                    min_timesteps_per_batch,
                ),
            )
            processes.append(p)
            p.start()

        # Wait for all workers to finish
        for p in processes:
            p.join()

        # Convert shared paths list back to a regular list
        paths = list(paths)
        timesteps_this_batch = timesteps_counter.value
        sample_trajectories.average_path_length = np.mean(
            [get_pathlength(path) for path in paths]
        )

        print(
            f"\nCollected timesteps: {timesteps_this_batch}/{min_timesteps_per_batch}"
        )
        print(f"Average path length: {sample_trajectories.average_path_length}")

    else:
        timesteps_this_batch = 0
        paths = []
        while timesteps_this_batch < min_timesteps_per_batch:

            # collect rollout
            path = sample_trajectory(env, policy, max_path_length, render, render_mode)
            paths.append(path)

            # count steps
            timesteps_this_batch += get_pathlength(path)
            print(
                "At timestep:    ",
                timesteps_this_batch,
                "/",
                min_timesteps_per_batch,
                end="\r",
            )

    return paths, timesteps_this_batch


sample_trajectories.average_path_length = None


def worker_sample_trajectories(
    worker_id,
    env_id,
    policy,
    max_path_length,
    render,
    render_mode,
    lock,
    timesteps_counter,
    paths,
    min_timesteps_per_batch,
):
    # Create a new environment instance
    worker_env = gym.make(env_id)

    # Have to reinstantiate the policy in the worker
    worker_policy = policy.copy()
    worker_policy.to(ptu.device)

    while True:
        # Sample a trajectory
        path = sample_trajectory(
            worker_env, worker_policy, max_path_length, render, render_mode
        )

        # Get the length of the trajectory
        path_length = get_pathlength(path)

        # Synchronize access to shared variables
        with lock:
            # Check if we've already collected enough timesteps
            if timesteps_counter.value >= min_timesteps_per_batch:
                break  # Exit the loop if we've collected enough data
            # Add the trajectory to the shared paths list
            paths.append(path)
            # Update the shared timesteps counter
            timesteps_counter.value += path_length

            # Optionally, print progress
            # print(f"Worker {worker_id}: Collected {timesteps_counter.value}/{min_timesteps_per_batch} timesteps", end='\r')

    # Close the worker's environment
    worker_env.close()


def sample_n_trajectories(
    env, policy, ntraj, max_path_length, render=False, render_mode=("rgb_array")
):
    # TODO: get this from hw1
    """
    Collect ntraj rollouts.
    """
    paths = []
    for i in range(ntraj):
        # collect rollout
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)

    return paths


############################################
############################################


def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
    Take info (separate arrays) from a single rollout
    and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }


def convert_listofrollouts(paths):
    """
    Take a list of rollout dictionaries
    and return separate arrays,
    where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return (
        observations,
        actions,
        next_observations,
        terminals,
        concatenated_rewards,
        unconcatenated_rewards,
    )


############################################
############################################


def get_pathlength(path):
    return len(path["reward"])


def normalize(data, mean, std, eps=1e-8):
    return (data - mean) / (std + eps)


def unnormalize(data, mean, std):
    return data * std + mean


def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp)  # (num data points, dim)

    # mean of data
    mean_data = np.mean(data, axis=0)

    # if mean is 0,
    # make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    # width of normal distribution to sample noise from
    # larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(
            data[:, j]
            + np.random.normal(0, np.absolute(std_of_noise[j]), (data.shape[0],))
        )

    return data
