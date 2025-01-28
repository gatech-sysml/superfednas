import abc
import collections
from collections import OrderedDict
from typing import Any, Collection, Dict, Optional

import attr
import torch
import copy
import fedml_api.dpftrl.tree_aggregation as pyp

def _check_momentum(m: float):
  if m < 0 or m >= 1:
    raise ValueError('Momenum should be in [0, 1), but got {}'.format(m))

class ServerOptimizerBase(metaclass=abc.ABCMeta):
  """Base class establishing interface for server optimizer."""

  @abc.abstractmethod
  def model_update(self, state: Dict[str, Any],
                   grad: dict,
                   round_idx: int) -> Dict[str, Any]:
    """Returns optimizer states after modifying in-place the provided `weight`.

    Args:
      state: optimizer state, usually defined/initialized in `init_state`.
      weight: model weights to be updated in this function.
      grad: gradients to update the model weights and optimizer states.
      round_idx: round/iteration index.

    Returns:
      Updated optimizer state.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def init_state(self) -> Dict[str, Any]:
    """Returns initialized optimizer state."""
    raise NotImplementedError

@attr.s(eq=False, frozen=True, slots=True)
class FTRLState(object):
  """Class defining state of the DP-FTRL optimizer.

  Attributes:
    init_weight: A Collection[tf.Tensor] defining the initial weight.
    sum_grad: A Collection[tf.Tensor] tracing the summation of gradient.
    dp_tree_state: A `  pyp.tree_aggregation.TreeState` tracking the state of the
      tree aggregatin noise for the additive in DP-FTRL algorithm.
    momentum_buffer:  A Collection[tf.Tensor] tracing the velocity in the
      momentum variant. Momentum is applied to the (noised) summation of
      gradients.
  """
  init_weight = attr.ib()
  sum_grad = attr.ib()
  dp_tree_state = attr.ib()
  momentum_buffer = attr.ib()

class DPFTRLMServerOptimizer(ServerOptimizerBase):
    """Momentum FTRL Optimizer with Tree aggregation for DP noise.

    There are two options of the tree aggregation algorithm:
    the baseline method `  pyp.tree_aggregation.TreeAggregator`, and the efficient
    method `  pyp.tree_aggregation.EfficientTreeAggregator` , which is controlled by
    flag `efficient_tree` in the constructor.
    """

    def __init__(self,
                learning_rate: float,
                momentum: float,
                noise_std: float,
                model_weight_specs: OrderedDict,
                efficient_tree: bool = True,
                use_nesterov: bool = False,
                noise_seed: Optional[int] = None):
        """Initialize the momemtum DPFTRL Optimizer."""

        _check_momentum(momentum)
        if use_nesterov and momentum == 0:
            raise ValueError('Use a positive momentum for Nesterov')

        self.lr = learning_rate
        self.momentum = momentum
        self.model_weight_specs = model_weight_specs
        self.use_nesterov = use_nesterov

        random_generator =  pyp.GaussianNoiseGenerator(
            noise_std, model_weight_specs, noise_seed)

        self.noise_generator =  pyp.EfficientTreeAggregator(
            value_generator=random_generator)
            
    def model_update(self, state: FTRLState, grad: OrderedDict, round_idx: int) -> FTRLState:
        """Returns optimizer state after one step update."""
        init_weight, sum_grad, dp_tree_state, momentum_buffer = (
            state.init_weight, state.sum_grad, state.dp_tree_state,
            state.momentum_buffer)
        for k in grad.keys():
            sum_grad[k] += grad[k]
        cumsum_noise, dp_tree_state = self.noise_generator.get_cumsum_and_update(dp_tree_state)
        noised_sum_grad = copy.deepcopy(sum_grad)
        for k in sum_grad.keys():
            noised_sum_grad[k] - cumsum_noise[k]
        for k in noised_sum_grad.keys():
            momentum_buffer[k] = self.momentum * momentum_buffer[k] + noised_sum_grad[k]
        if self.use_nesterov:
        # The Nesterov implementation mimics the implementation of
        # `tf.keras.optimizers.SGD`. The forecasted weight is used to generate
        # gradient in momentum buffer (velocity).
            delta_w = copy.deepcopy(momentum_buffer)
            for k in momentum_buffer.keys():
                delta_w[k] = self.momentum * delta_w[k] + noised_sum_grad[k]
        else:
            delta_w = momentum_buffer
            
        # Different from a conventional SGD step, FTRL use the initial weight w0
        # and (momementum version of) the gradient sum to update the model weight.
        output = copy.deepcopy(init_weight)
        for k in delta_w.keys():
            output[k] = init_weight[k] + self.lr * delta_w[k]
        state = FTRLState(
            init_weight=init_weight,
            sum_grad=sum_grad,
            dp_tree_state=dp_tree_state,
            momentum_buffer=momentum_buffer)
        return output, state

    def _zero_state(self):
        zstate = OrderedDict()
        for k, v in self.model_weight_specs.items():
            zstate[k] = torch.zeros(v.shape, dtype=torch.float32)
        return zstate


    def init_state(self, weight) -> FTRLState:
        """Returns initialized optimizer and tree aggregation states."""
        return FTRLState(
            init_weight=weight,
            sum_grad=self._zero_state(),
            dp_tree_state=self.noise_generator.init_state(),
            momentum_buffer=self._zero_state())

    def restart_dp_tree(self, weight) -> FTRLState:
        """Returns a reinitialized state based on the current model weights."""
        return FTRLState(
            init_weight=weight,
            sum_grad=self._zero_state(),
            dp_tree_state=self.noise_generator.init_state(),
            momentum_buffer=self._zero_state())
