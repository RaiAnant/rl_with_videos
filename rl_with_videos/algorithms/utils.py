from copy import deepcopy


def create_SAC_algorithm(variant, *args, **kwargs):
    from .sac import SAC

    algorithm = SAC(*args, **kwargs)

    return algorithm


def create_RLV_algorithm(variant, *args, **kwargs):
    from .rl_with_videos import RLV

    algorithm = RLV(*args, **kwargs)

    return algorithm

def create_RLVU_algorithm(variant, *args, **kwargs):
    from .rlvu import RLVU

    algorithm = RLVU(*args, **kwargs)

    return algorithm


ALGORITHM_CLASSES = {
    'SAC': create_SAC_algorithm,
    'RLV': create_RLV_algorithm,
    'RLVU': create_RLVU_algorithm,
}


def get_algorithm_from_variant(variant,
                               *args,
                               **kwargs):
    algorithm_params = variant['algorithm_params']
    algorithm_type = algorithm_params['type']
    algorithm_kwargs = deepcopy(algorithm_params['kwargs'])
    algorithm = ALGORITHM_CLASSES[algorithm_type](
        variant, *args, **algorithm_kwargs, **kwargs)

    return algorithm
