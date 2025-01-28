import torch
import torchvision
from torchvision import datasets
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import List, Tuple
from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler
from torchvision import datasets, transforms
import pytorch_lightning as pl

from sklearn.decomposition import PCA

# do
from torch.nn.utils import clip_grad_norm_

# from opacus import PrivacyEngine
# from opacus.data_loader import DPDataLoader
# from opacus.validators.module_validator import ModuleValidator
from tqdm import tqdm

import json

# do
import math
import sys

# from pvc import Dataset
from typing import List, Optional, Tuple

# do
import matplotlib.pyplot as plt


# TODO: clip norm


class TreeNode:
    def __init__(self, depth: int, value: float, efficient: bool):
        """
        implements the nodes of the tree. if the tree is efficient we will weigh
        the nodes by the weight method. adapted from opacus privacy engine.
        """
        self.depth = depth
        self.value = value
        self.efficient = efficient

    def update_value(self, indx, new_value):
        self.value[indx] += new_value

    def get_value(self, indx):
        """as indicated by the efficient implementation of tree aggregation
        we reweight it by 1 / (2 - 2^(-depth)) where depth starts from the root.
        ex. depth(root) = 0
        """
        if self.efficient:
            # print('dd', self.depth)
            # print('efficient test', float((1.0 / (2 - math.pow(2, -self.depth))) ** 0.5) * self.value)
            return ((1.0 / (2 - math.pow(2, -self.depth))) ** 0.5) * self.value[indx]
        else:
            return 1.0 * self.value[indx]


class TreeAggregation:
    """
    Correlated noise added to gradient sums from a private binary tree
    Efficient Implementation Paper (reduces variance in sum of gaussian noise values):
        Efficient Use of Differentially Private Binary Trees
        https://privacytools.seas.harvard.edu/files/privacytools/files/honaker.pdf
    Vanilla Implementation:
        Differential Privacy Under Continual Observation
        http://www.wisdom.weizmann.ac.il/~naor/PAPERS/continual_observation.pdf

    Args:
        std = standard deviation of gaussian noise
        efficient = decides which kind of tree aggregation to be used
        seed = added for reproducibility
    """

    def __init__(
        self,
        grad_sizes,
        std: float = 1.0,
        efficient: bool = False,
        seed: int = 0,
        mode: str = "prod",  # alternatives are prod/test
    ):
        self.efficient = efficient
        self.std = std
        self.seed = seed
        self.mode = mode
        self.grad_sizes = grad_sizes

        self.step = 1
        self.depth = 0  # TODO: increase depth appropriately
        self.max_nodes = 2 ** (self.depth)
        self.tree = []

        # reproducible experiments
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

    def add_to_tree_helper(
        self,
        gradient,
        noise_to_add,
        binary_repr_step,
        step,
        max_nodes,
        add_gradient,
    ):
        """
        adds noise to each node on the path from the root node to the leaf node.
        Also, returns correlated noise for this specific path to be added to the FTRL optimizer.

        Args:
            noise_to_add = torch Tensor (gaussian noise) to be added
            binary_repr_step = convert current step value to binary representation
            step = number of times this function has been called (correlated with maximum nodes in the system)
            max_nodes = maximum possible nodes in the tree (i.e. counted after assuming a complete binary tree,
                where each node except the leaf node have exactly two childen)

        Returns:
            correlated noise determined from the current tree structure & step number
        """
        if len(self.tree) == 0:
            # print('Root Added')
            noise_sum = [torch.zeros(grad_size) for grad_size in self.grad_sizes]
            self.tree.append(
                TreeNode(depth=self.depth, value=noise_to_add, efficient=self.efficient)
            )
            # print(len(self.tree), gradient, len(gradient))
            if add_gradient:
                for indx in range(len(self.grad_sizes)):
                    self.tree[0].update_value(indx, gradient[indx])

            for indx in range(len(self.grad_sizes)):
                noise_sum[indx] += self.tree[0].get_value(indx)
            return noise_sum

        # node_indx = 0
        binary_step_indx = 1

        # append noise & init noise_sum
        self.tree.append(
            TreeNode(depth=self.depth, value=noise_to_add, efficient=self.efficient)
        )
        noise_sum = [torch.zeros(grad_size) for grad_size in self.grad_sizes]

        # for indx in range(len(self.grad_sizes)):
        #    self.tree[step].update_value(indx, gradient[indx])

        # get back value
        if binary_repr_step[0] == "1":  # test if left node
            for indx in range(len(self.grad_sizes)):
                noise_sum[indx] += self.tree[step].get_value(indx)

        steps = [step]
        while step >= 0 and binary_step_indx < len(binary_repr_step):
            left_indx = int((step - 1) / 2)
            if step % 2 == 1:  # is left child
                step = int((step - 1) / 2)
            else:  # is right node
                step = int((step - 2) / 2)
            steps.append(step)
            if add_gradient:
                for indx in range(len(self.grad_sizes)):
                    self.tree[step].update_value(indx, gradient[indx])

            # get value
            if binary_repr_step[binary_step_indx] == "1" and left_indx < len(self.tree):
                for indx in range(len(self.grad_sizes)):
                    noise_sum[indx] += self.tree[left_indx].get_value(indx)

            binary_step_indx += 1

        # print(steps)
        return noise_sum

    def print_tree(self):
        print("------------------------Tree Starts------------------------")
        for indx in range(len(self.tree)):
            print(
                "(" + str(indx + 1) + "," + str(self.tree[indx].value[0].item()) + ")",
                end=" ",
            )
        print("\n------------------------Tree Ends------------------------")

    def add_to_tree_and_get_sum(
        self,
        gradient: torch.Tensor,
        test_noise: Optional[torch.Tensor] = None,
        add_gradient: bool = False,
    ) -> None:
        """
        add the new value to each node along the path

        Args:
            test_noise = torch.normal(mean=0.0, std=self.std, generator=torch.Generator())

        Returns:
            None
        """
        if self.mode == "prod":
            noise_to_add = []
            for indx in range(len(self.grad_sizes)):
                # print('size', self.grad_sizes[indx])
                noise_to_add.append(
                    torch.normal(
                        mean=0.0,
                        std=self.std,
                        size=self.grad_sizes[indx],
                        # generator=self.generator
                    )
                )
        elif self.mode == "test":
            noise_to_add = test_noise

        noise_sum = self.add_to_tree_helper(
            gradient,
            noise_to_add,
            np.binary_repr(self.step)[::-1],
            self.step - 1,
            self.max_nodes + 1,
            add_gradient,
        )

        self.step += 1
        # print('metric', self.step, self.max_nodes, self.depth)
        if self.step > self.max_nodes:
            self.depth += 1
            self.max_nodes += 2 ** (self.depth)

        return noise_sum


from typing import List

# do
import torch
from torch.optim.optimizer import Optimizer, required


class FTRLM(Optimizer):
    """
    implements FTRL optimizer to be used with Opacus or alternative noise generation
    mechanism with (optional) momentum

    """

    def __init__(
        self,
        device,
        params,  # TODO: add tree aggregator
        model_param_sizes: List[torch.Size] = [(1,)],
        lr: float = 0.0,
        momentum: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        noise_std: float = 0.0,
        max_grad_norm: float = 100.0,
        seed: int = 0,
        efficient: bool = False,
        is_PCA_enabled: bool = False,
        PCA_variance: float = 0.95,
    ):
        """
        Args:
            params = parameters for torch optimizer
            lr = learning rate
            momentum = # TODO: add implementation
            nesterov = # TODO: add implementation
            dampening = # # TODO: add implementation
            noise_std = amount of noise to be added to teh
        """
        self.device = device
        # sanity checks for parameters
        if lr <= 0.0:
            raise ValueError("Invalid Learning Rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid Momentum Parameter: {}".format(momentum))
        if isinstance(max_grad_norm, float) and max_grad_norm <= 0:
            raise ValueError(
                "max_grad_norm = {} is not a valid value. Please provide a float > 0.".format(
                    max_grad_norm
                )
            )

        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.noise_std = noise_std
        self.efficient = efficient
        self.is_PCA_enabled = is_PCA_enabled
        self.PCA_variance = PCA_variance

        self.tree_aggregator = TreeAggregation(
            grad_sizes=model_param_sizes,
            std=self.noise_std,
            efficient=self.efficient,
            seed=self.seed,
        )

        # TODO: add weight decay
        # if not 0.0 < weight_decay:
        #    raise ValueError("Invalid Weight Decay Parameter: {}".format(weight_decay))

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(
            lr=lr, dampening=dampening, momentum=momentum, nesterov=nesterov
        )

        super().__init__(params, defaults)

    def clip_gradient(self, grad):  # TODO: add per-example clipping
        # print('Sanity Check', grad.shape)
        if torch.norm(grad) > self.max_grad_norm:
            return (grad * self.max_grad_norm) / float(torch.norm(grad))
        return grad

    def step(self, closure=None):
        """performs one step of the DP-FTRL algorithm.

        Args:
            closure (callable, optional): a closure that reevaluates
            the model and returns the loss
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        gradients = []
        noisy_gradients = self.tree_aggregator.add_to_tree_and_get_sum(
            gradients, add_gradient=False
        )

        for group in self.param_groups:
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            lr = group["lr"]
            for p, noise in zip(group["params"], noisy_gradients):
                if p.grad is None:
                    continue

                d_p = p.grad
                d_p = self.clip_gradient(d_p)
                param_state = self.state[p]
                if len(param_state) == 0:
                    param_state["grad_sum"] = torch.zeros_like(d_p)
                    param_state["momentum"] = torch.zeros_like(p)
                    param_state["initial_model"] = torch.zeros_like(p)
                    param_state["initial_model"].add_(p)  # only add initial model

                initial_model, grad_sum = (
                    param_state["initial_model"],
                    param_state["grad_sum"],
                )

                grad_sum.add_(d_p)
                # PCA
                # noisy_gradient = noisy_gradients[indx]
                """if self.is_PCA_enabled:
                    pca = PCA(self.PCA_variance)
                    dimension2 = 1
                    gradient_shape = noisy_gradient.shape
                    for i in range(len(noisy_gradient.shape)):
                        if i != 0:
                            dimension2 *= gradient_shape[i]
                    noisy_gradient_2D = noisy_gradient.reshape((gradient_shape[0], dimension2))
                    noisy_transformed_gradient = pca.fit_transform(noisy_gradient_2D)"""
                noise = noise.to(self.device)
                param_state["momentum"].mul_(momentum).add_(
                    grad_sum + noise, alpha=(1 - dampening)
                )

                # Nesterov
                if nesterov:
                    delta_w = (grad_sum + noise).add(
                        param_state["momentum"], alpha=momentum
                    )
                else:
                    delta_w = param_state["momentum"]

                # p = initial_model
                with torch.no_grad():
                    p.copy_(initial_model - delta_w * lr)

        return loss


# do
import unittest
from copy import deepcopy


class FTRLMTest(unittest.TestCase):
    # init
    def __init__(self):
        pass
        # add multiple tests

    # utils
    def change_lr(self, opt, new_lr):
        for group in opt.param_groups:
            group["lr"] = new_lr

    def get_lr(self, opt):
        for group in opt.param_groups:
            return group["lr"]

    def get_momentum(self, opt):
        for group in opt.param_groups:
            return group["momentum"]

    # tests
    def test_lr_deterministic(self):
        param = torch.tensor([0.0], requires_grad=True)
        param.grad = torch.tensor([1.0])
        self.optimizer = FTRLM([param], lr=1, noise_std=0.0, max_grad_norm=10.0)
        # self.optimizer.set_param_groups([param], lr=1)

        # output = 0 - 1.0 * 1.0 = -1.0
        self.assertAlmostEqual(self.get_lr(self.optimizer), 1.0)
        self.optimizer.step()  # 1st step
        print(param.item())
        self.assertAlmostEqual(param.item(), -1.0, delta=1e-5)

        # output = 0 - 1.5 * (1.0 + 1.0) = -3.0
        self.change_lr(self.optimizer, 1.5)  #
        self.assertAlmostEqual(self.get_lr(self.optimizer), 1.5)
        self.optimizer.step()
        print(param.item())
        self.assertAlmostEqual(param.item(), -3.0, delta=1e-5)

    def test_momentum(self):
        param = torch.zeros((784, 10), requires_grad=True)
        shapes = [param.shape]
        # shapes = [p.shape for p in param]
        print(shapes)
        param.grad = torch.ones_like(param)
        self.optimizer = FTRLM(
            [param],
            lr=0.1,
            model_param_sizes=shapes,
            noise_std=0.0,
            momentum=0.9,
            max_grad_norm=90.0,
        )

        self.assertAlmostEqual(self.get_lr(self.optimizer), 0.1)
        self.assertAlmostEqual(self.get_momentum(self.optimizer), 0.9)
        for epoch in range(2):
            self.optimizer.step()
        output = torch.flatten(param)
        print(output)
        self.assertTrue(
            torch.allclose(
                output,
                torch.Tensor(
                    [-0.29 * np.ones_like(p.detach().numpy()) for p in output]
                ),
            )
        )

    def test_ftrl_similar_sgd(self):  # without nosie should perform simlarly to SGD
        param_optimizer = torch.normal(
            mean=0.0, std=1.0, size=(784, 10), requires_grad=True
        )
        # param_optimizer = torch.zeros((784, 10))
        param_optimizer.grad = torch.normal(
            mean=0.0, std=1.0, size=(784, 10), requires_grad=True
        )
        # print('init grad', param_optimizer.grad)
        shapes = [param_optimizer.shape]
        param_sgd = deepcopy(param_optimizer)
        param_sgd.grad = deepcopy(param_optimizer.grad)
        self.optimizer = FTRLM(
            [param_optimizer],
            model_param_sizes=shapes,
            lr=0.01,
            noise_std=0.0,
            max_grad_norm=500.0,
        )
        self.sgd = torch.optim.SGD([param_sgd], lr=0.01)

        for epoch in range(3):
            self.optimizer.step()
            self.sgd.step()
        print((param_optimizer), (param_sgd))
        self.assertTrue(
            torch.allclose(torch.flatten(param_optimizer), torch.flatten(param_sgd))
        )

    def test_ftrl_similar_sgd_momentum(
        self,
    ):  # without nosie should perform simlarly to SGD
        param_optimizer = torch.normal(
            mean=0.0, std=1.0, size=(784, 10), requires_grad=True
        )
        # param_optimizer = torch.zeros((784, 10))
        param_optimizer.grad = torch.normal(
            mean=0.0, std=1.0, size=(784, 10), requires_grad=True
        )
        # print('init grad', param_optimizer.grad)
        shapes = [param_optimizer.shape]
        param_sgd = deepcopy(param_optimizer)
        param_sgd.grad = deepcopy(param_optimizer.grad)
        self.optimizer = FTRLM(
            [param_optimizer],
            model_param_sizes=shapes,
            lr=0.01,
            noise_std=0.0,
            max_grad_norm=500.0,
            momentum=0.9,
        )
        self.sgd = torch.optim.SGD([param_sgd], lr=0.01, momentum=0.9)

        for epoch in range(3):
            self.optimizer.step()
            self.sgd.step()
        print((param_optimizer), (param_sgd))
        self.assertTrue(
            torch.allclose(torch.flatten(param_optimizer), torch.flatten(param_sgd))
        )

if __name__ == '__main__':
    test = FTRLMTest()
    test.test_lr_deterministic()  # DONE
    test.test_momentum()
    test.test_ftrl_similar_sgd()
    test.test_ftrl_similar_sgd_momentum()

####DATALOADERS####

####NET####


