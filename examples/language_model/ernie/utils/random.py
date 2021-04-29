import paddle
import contextlib

MODEL_PARALLEL_RNG = 'model_parallel_rng'


class RNGStatesTracker:
    """
    Tracker the RNG states.
    """

    def __init__(self):
        # Map from name to the rng state.
        self.states_ = {}
        self.seeds_ = set()

    def add(self, name, seed):
        if seed in self.seeds_:
            raise ValueError('seed {} already exists'.format(seed))
        self.seeds_.add(seed)
        if name in self.states_:
            raise ValueError('state {} already exists'.format(name))
        orig_rng_state = paddle.get_cuda_rng_state()
        paddle.seed(seed)
        self.states_[name] = paddle.get_cuda_rng_state()
        paddle.set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def rng_state(self, name=MODEL_PARALLEL_RNG):
        if name not in self.states_:
            raise ValueError('state {} does not exist'.format(name))
        orig_cuda_rng_state = paddle.get_cuda_rng_state()
        paddle.set_cuda_rng_state(self.states_[name])
        try:
            yield
        finally:
            self.states_[name] = paddle.get_cuda_rng_state()
            paddle.set_cuda_rng_state(orig_cuda_rng_state)


RNG_STATE_TRACKER = RNGStatesTracker()


def get_rng_state_tracker():
    return RNG_STATE_TRACKER
