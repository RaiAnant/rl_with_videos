from copy import deepcopy

from rl_with_videos.replay_pools import video_replay_pool

from . import (
    action_free_replay_pool,
    simple_replay_pool,
    extra_policy_info_replay_pool,
    union_pool,
    trajectory_replay_pool,
    active_replay_pool)


'''
action_free_replay_pool:  ActionFreeReplayPool
replay_pool: SimpleReplayPool
'''


POOL_CLASSES = {
    'ActionFreeReplayPool': action_free_replay_pool.ActionFreeReplayPool,
    'SimpleReplayPool': simple_replay_pool.SimpleReplayPool,
    'ActiveReplayPool': active_replay_pool.ActiveReplayPool,
    'TrajectoryReplayPool': trajectory_replay_pool.TrajectoryReplayPool,
    'ExtraPolicyInfoReplayPool': (
        extra_policy_info_replay_pool.ExtraPolicyInfoReplayPool),
    'UnionPool': union_pool.UnionPool,
    'VideoReplayPool': video_replay_pool.VideoReplayPool,
}

DEFAULT_REPLAY_POOL = 'SimpleReplayPool'


def get_replay_pool_from_variant(variant, env, *args, **kwargs):
    replay_pool_params = variant['replay_pool_params']
    replay_pool_type = replay_pool_params['type']
    replay_pool_kwargs = deepcopy(replay_pool_params['kwargs'])

    replay_pool = POOL_CLASSES[replay_pool_type](
        *args,
        observation_space=env.observation_space,
        action_space=env.action_space,
        **replay_pool_kwargs,
        **kwargs)

    return replay_pool
