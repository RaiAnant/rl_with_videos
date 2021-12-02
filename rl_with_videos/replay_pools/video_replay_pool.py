from gym.spaces import Dict

from .flexible_replay_pool import FlexibleReplayPool
from .simple_replay_pool import normalize_observation_fields

class VideoReplayPool(FlexibleReplayPool):
    def __init__(self,
                 observation_space,
                 action_space,
                 data_path=None,
                 *args,
                 extra_fields=None,
                 remove_rewards=False,
                 use_ground_truth_actions=False,
                 max_demo_length=-1,
                 **kwargs):
        extra_fields = extra_fields or {}
#        action_space = environment.action_space
#        assert isinstance(observation_space, Dict), observation_space

#        self._environment = environment
        self._observation_space = observation_space
        self._action_space = action_space
        print("self._observation_space", self._observation_space)

        # observation_fields = normalize_observation_fields(observation_space)
        # # It's a bit memory inefficient to save the observations twice,
        # # but it makes the code *much* easier since you no longer have
        # # to worry about termination conditions.
        # observation_fields.update({
        #     'next_' + key: value
        #     for key, value in observation_fields.items()
        # })

        fields = {
            **{
                'sequences': {
                    'shape': (30, 6,),
                    'dtype': 'float32'
                },
                'done': {
                    'shape': (1, ),
                    'dtype': 'float32'
                },
            }
        }

        super(VideoReplayPool, self).__init__(
            *args, fields_attrs=fields, **kwargs)
        print("about to load replay pool")
        self.load_experience(data_path)
        print("loaded experience of size:", self.size) 

        if max_demo_length != -1 and max_demo_length < self.fields['observations'].shape[0] and max_demo_length < self._size:
            print("going from size {} or {} to size {}".format(self.fields['observations'].shape[0], self._size, max_demo_length))
            for k in self.fields.keys():
                self.fields[k] = self.fields[k][self._size-max_demo_length:self._size]
            self._size = max_demo_length


    """ The action-free replay pool should not be added to during runtime
    This removes the methods that were inherited from FlexibleReplayPool
    """
    def add_sample(self, sample):
        raise NotImplementedError

    def add_path(self, path):
        raise NotImplementedError