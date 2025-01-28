import abc
import collections
from typing import Any, Callable, Collection, NamedTuple, Optional, Tuple, Union

import torch 


class ValueGenerator(metaclass=abc.ABCMeta):
  """Base class establishing interface for stateful value generation.

  A `ValueGenerator` maintains a state, and each time `next` is called, a new
  value is generated and the state is advanced.
  """

  @abc.abstractmethod
  def initialize(self):
    """Makes an initialized state for the ValueGenerator.

    Returns:
      An initial state.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def next(self, state):
    """Gets next value and advances the ValueGenerator.

    Args:
      state: The current state.

    Returns:
      A pair (value, new_state) where value is the next value and new_state
        is the advanced state.
    """
    raise NotImplementedError

class GaussianNoiseGenerator(ValueGenerator):
    """Gaussian noise generator with counter as pseudo state.

    Produces i.i.d. spherical Gaussian noise at each step shaped according to a
    nested structure of model weights.
    """

    # pylint: disable=invalid-name
    _GlobalState = collections.namedtuple('_GlobalState', ['seeds', 'stddev'])

    def __init__(self,
                noise_std: float,
                weight_shapes: Collection[torch.Tensor],
                seed: Optional[int] = None):
        """Initializes the GaussianNoiseGenerator.

        Args:
        noise_std: The standard deviation of the noise.
        weighted_shapes: A nested structure of tensors specifying the shape of the
            noise to generate.
        seed: An optional integer seed. If None, generator is seeded from the
            clock.
        """
        self._noise_std = noise_std
        self._specs = weight_shapes
        self._seed = seed

    def initialize(self):
        """Makes an initial state for the GaussianNoiseGenerator.

        Returns:
        A named tuple of (seeds, stddev).
        """
        if self._seed is None:
            #default seed of 7
            self._seed = 7
        return self._GlobalState(
            torch.tensor(self._seed, dtype=torch.int64, shape=(2,)),
            torch.tensor(self._noise_std, dtype=torch.float32))

    def next(self, state):
        """Gets next value and advances the GaussianNoiseGenerator.

        Args:
            state: The current state (seed, noise_std).

        Returns:
            A tuple of (sample, new_state) where sample is a new sample and new_state
                is the advanced state (seed+1, noise_std).
        """
        # Flatten the structure of weights
        flat_structure = [spec.shape for spec in self._specs]
        flat_seeds = [state.seeds + i for i in range(len(flat_structure))]
        
        # Generate Gaussian noise for each spec using the seeds
        def _get_noise(shape, seed):
            generator = torch.Generator()
            generator.manual_seed(seed)
            return torch.normal(mean=0.0, std=state.stddev, size=shape, generator=generator)
        
        # Map seeds to the weight shapes
        nest_noise = [
            _get_noise(shape, seed) for shape, seed in zip(flat_structure, flat_seeds)
        ]

        return nest_noise, self._GlobalState(flat_seeds[-1] + 1, state.stddev)

    def make_state(self, seeds: torch.Tensor, stddev: torch.Tensor):
        """Returns a new named tuple of (seeds, stddev)."""
        seeds = seeds.view(2)  # Ensure the shape is (2,)
        return self._GlobalState(
            seeds.to(dtype=torch.int64), stddev.to(dtype=torch.float32)
        )

    

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Efficient Tree Aggregator Implementation Port over
class TreeState(NamedTuple):
    """Class defining state of the tree.

    Attributes:
        level_buffer: A `tf.Tensor` saves the last node value of the left child
        entered for the tree levels recorded in `level_buffer_idx`.
        level_buffer_idx: A `tf.Tensor` for the tree level index of the
        `level_buffer`.  The tree level index starts from 0, i.e.,
        `level_buffer[0]` when `level_buffer_idx[0]==0` recorded the noise value
        for the most recent leaf node.
    value_generator_state: State of a stateful `ValueGenerator` for tree node.
    """
    level_buffer: torch.Tensor
    level_buffer_idx: torch.Tensor
    value_generator_state: Any

def get_step_idx(state: TreeState) -> torch.Tensor:
    """Returns the current leaf node index based on `TreeState.level_buffer_idx`."""
    step_idx = torch.tensor(-1, dtype=torch.int32)
    for i in range(len(state.level_buffer_idx)):
        step_idx += 2 ** state.level_buffer_idx[i]
    return step_idx

class StatelessValueGenerator(ValueGenerator):
  """A wrapper for stateless value generator that calls a no-arg function."""

  def __init__(self, value_fn):
    """Initializes the StatelessValueGenerator.

    Args:
      value_fn: The function to call to generate values.
    """
    self.value_fn = value_fn

  def initialize(self):
    """Makes an initialized state for the StatelessValueGenerator.

    Returns:
      An initial state (empty, because stateless).
    """
    return ()

  def next(self, state):
    """Gets next value.

    Args:
      state: The current state (simply passed through).

    Returns:
      A pair (value, new_state) where value is the next value and new_state
        is the advanced state.
    """
    return self.value_fn(), state

class EfficientTreeAggregator():
    """Efficient tree aggregator to compute accumulated noise.

    This class implements the efficient tree aggregation algorithm based on
    Honaker 2015 "Efficient Use of Differentially Private Binary Trees".
    The noise standard deviation for a node at depth d is roughly
    `sigma * sqrt(2^{d-1}/(2^d-1))`. which becomes `sigma / sqrt(2)` when
    the tree is very tall.

    Example usage:
        random_generator = GaussianNoiseGenerator(...)
        tree_aggregator = EfficientTreeAggregator(random_generator)
        state = tree_aggregator.init_state()
        for leaf_node_idx in range(total_steps):
        assert leaf_node_idx == get_step_idx(state))
        noise, state = tree_aggregator.get_cumsum_and_update(state)

    Attributes:
        value_generator: A `ValueGenerator` or a no-arg function to generate a noise
        value for each tree node.
    """

    def __init__(self, value_generator: Union[ValueGenerator, Callable[[], Any]]):
        """Initialize the aggregator with a noise generator.

        Args:
        value_generator: A `ValueGenerator` or a no-arg function to generate a
            noise value for each tree node.
        """
        if isinstance(value_generator, ValueGenerator):
            self.value_generator = value_generator
        else:
            self.value_generator = StatelessValueGenerator(value_generator)

    def _get_init_state(self, value_generator_state):
        """Returns initial buffer for `TreeState`."""
        level_buffer_idx = torch.tensor([0], dtype=torch.int32)

        new_val, value_generator_state = self.value_generator.next(value_generator_state)

        if isinstance(new_val, dict): 
            level_buffer_structure = {k: [torch.tensor(v, dtype=torch.float32)] for k, v in new_val.items()}
        else:  
            level_buffer_structure = [torch.tensor(new_val, dtype=torch.float32)]

        level_buffer = {k: torch.stack(v) for k, v in level_buffer_structure.items()} if isinstance(level_buffer_structure, dict) else torch.stack(level_buffer_structure)
        
        return TreeState(
            level_buffer=level_buffer,
            level_buffer_idx=level_buffer_idx,
            value_generator_state=value_generator_state)

    def init_state(self) -> TreeState:
        """Returns initial `TreeState`.

        Initializes `TreeState` for a tree of a single leaf node: the respective
        initial node value in `TreeState.level_buffer` is generated by the value
        generator function, and the node index is 0.

        Returns:
        An initialized `TreeState`.
        """
        value_generator_state = self.value_generator.initialize()
        return self._get_init_state(value_generator_state)

    def reset_state(self, state: TreeState) -> TreeState:
        """Returns reset `TreeState` after restarting a new tree."""
        return self._get_init_state(state.value_generator_state)


    def _get_cumsum(self, state: TreeState) -> torch.Tensor:
        """Returns weighted cumulative sum of noise based on `TreeState`."""
        # Note that the buffer saved recursive results of the weighted average of
        # the node value (v) and its two children (l, r), i.e., node = v + (l+r)/2.
        # To get unbiased estimation with reduced variance for each node, we have to
        # reweight it by 1/(2-2^{-d}) where d is the depth of the node.
        level_weights = 1. / (2. - torch.pow(0.5, state.level_buffer_idx.float()))

        def _weighted_sum(buffer):
            expand_shape = [len(level_weights)] + [1] * (buffer.dim() - 1)
            weighted_buffer = buffer * level_weights.view(*expand_shape)
            return torch.sum(weighted_buffer, dim=0)

        if isinstance(state.level_buffer, dict):
            return {k: _weighted_sum(v) for k, v in state.level_buffer.items()}
        else: 
            return _weighted_sum(state.level_buffer)

    def get_cumsum_and_update(self, state: TreeState) -> Tuple[torch.Tensor, TreeState]:
        """Returns tree aggregated noise and updates `TreeState` for the next step."""
        cumsum = self._get_cumsum(state)

        level_buffer_idx = state.level_buffer_idx
        level_buffer = state.level_buffer
        value_generator_state = state.value_generator_state

        new_level_buffer = [torch.zeros_like(buf) for buf in level_buffer]
        new_level_buffer_idx = []

        level_idx = 0
        new_value, value_generator_state = self.value_generator.next(value_generator_state)

        while level_idx < len(level_buffer_idx) and level_idx == level_buffer_idx[level_idx]:
            node_value, value_generator_state = self.value_generator.next(value_generator_state)
            new_value = [
                0.5 * (level[level_idx] + new_val) + node_val
                for level, node_val, new_val in zip(level_buffer, node_value, new_value)
            ]
            level_idx += 1
        write_buffer_idx = 0
        new_level_buffer_idx.append(level_idx)

        #do we need to concatenate rather then append to start?
        # Overwrite val at write_buffer_idx in each buffer
        for buf, val in zip(new_level_buffer, new_value):
            buf[write_buffer_idx] = val
        write_buffer_idx += 1

        for idx in range(level_idx, len(level_buffer_idx)):
            new_level_buffer_idx.append(level_buffer_idx[idx])
            for old_buf in level_buffer:
                new_level_buffer[write_buffer_idx] = old_buf[idx]
            write_buffer_idx += 1

        new_level_buffer_idx = torch.tensor(new_level_buffer_idx, dtype=torch.int32)
        new_state = TreeState(
            level_buffer=new_level_buffer,
            level_buffer_idx=new_level_buffer_idx,
            value_generator_state=value_generator_state,
        )
        return cumsum, new_state