from collections import defaultdict

import numpy as np

from .base_sampler import BaseSampler
from .simple_sampler import SimpleSampler


class SimpleSamplerVU(SimpleSampler):

    def _init_video_data_pool(self, video_replay_pool):
        # impqort pdb; pdb.set_trace() #breakpoint
        self.video_replay_pool = video_replay_pool

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        action = self.policy.actions_np([
            self.env.convert_to_active_observation(
                self._current_observation)[None]
        ])[0]

        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1
        terminal = terminal or self._path_length >= self._max_path_length

        processed_sample = self._process_observations(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            info=info,
        )

        for key, value in processed_sample.items():
            self._current_path[key].append(value)

        if terminal or self._path_length >= self._max_path_length:

            # self.video_replay_pool.add_sample(self._current_path)  # will need to be changed

            last_path = {
                field_name: np.array(values)
                for field_name, values in self._current_path.items()
            }

            self.pool.add_path(last_path)
            # import pdb; pdb.set_trace() #breakpoint
            self.video_replay_pool.add_sample({'sequences' :last_path['observations'][-30:], 'done': np.array([1 if terminal else 0])})
            self._last_n_paths.appendleft(last_path)

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self.policy.reset()
            self._current_observation = None
            self._path_length = 0
            self._path_return = 0
            self._current_path = defaultdict(list)

            self._n_episodes += 1
        else:
            self._current_observation = next_observation
        # video_pool.add(self._current_path)
        return next_observation, reward, terminal, info
