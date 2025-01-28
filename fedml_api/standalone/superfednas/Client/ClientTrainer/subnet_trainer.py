from fedml_api.standalone.superfednas.Client.ClientTrainer.client_trainer import ClientTrainer
from fedml_api.standalone.superfednas.elastic_nn.TCN.word_cnn.utils import (
    get_batch,
)
import numpy as np
import torch
import logging
from torch import nn
import torch.nn.functional as F
# from fedml_api.dpftrl.dp_ftrl import FTRLM


class SubnetTrainer(ClientTrainer):
    def __init__(self, model, device, args, teacher_model=None):
        super(SubnetTrainer, self).__init__(model, device, args, teacher_model)
        self.test_model = model
        self.alpha = args.feddyn_alpha

    def set_alpha(self, alpha):
        self.alpha = alpha
        
    def train(self, lr, local_ep, **kwargs):
        if self.teacher_model is not None:
            self.teacher_model.to(self.device)
            self.teacher_model.eval()

        self.client_model.to(self.device)
        self.client_model.train()
        if not self.args.use_bn:
            for m in self.client_model.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                    m.running_mean.requires_grad = False
                    m.running_var.requires_grad = False
                    with torch.no_grad():
                        m.weight.fill_(1)
                        m.bias.fill_(0)
                        m.running_mean.fill_(0)
                        m.running_var.fill_(1)

        # train and update
        criterion = nn.CrossEntropyLoss().to(self.device)
        cur_wd = self.args.wd
        if (
            self.client_model.is_max_net(self.client_model.model_config)
            and self.args.largest_subnet_wd
        ):
            cur_wd = self.args.largest_subnet_wd

        if self.args.mod_wd_dyn:
            cur_wd += self.alpha
        model_params = filter(lambda p: p.requires_grad, self.client_model.parameters())

        if self.args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model_params, lr=lr, weight_decay=cur_wd,)
        elif self.args.client_optimizer == "ftrl" :
            # param_shapes = [p.shape for p in self.client_model.parameters()]
            param_shapes = []
            for p in self.client_model.parameters():
                if p.requires_grad:
                    param_shapes.append(p.shape)
            # self.args 
            # optimizer = FTRLM( # noqa
            #             device = self.args.device,
            #             params=model_params,
            #             model_param_sizes=param_shapes,
            #             lr=lr,
            #             momentum=0.9,
            #             nesterov=True,
            #             noise_std=(self.args.noise_multiplier / 1000),
            #             max_grad_norm=self.args.max_norm,
            #             seed=self.args.init_seed,
            #             efficient=True,
            #         )
            if self.client_model.tree_aggregator != None:
                optimizer.tree_aggregator = self.client_model.tree_aggregator
            else:
                self.client_model.tree_aggregator = optimizer.tree_aggregator

        else:
            optimizer = torch.optim.Adam(
                model_params, lr=lr, weight_decay=cur_wd, amsgrad=True,
            )

  

        epoch_loss = []
        for epoch in range(local_ep if local_ep is not None else self.args.epochs):
            batch_loss = []
            if self.args.dataset == 'ptb':
                for batch_idx, i in enumerate(range(0, self.local_training_data.size(1) - 1, self.args.validseqlen)):
                    if i + self.args.seq_len - self.args.validseqlen >= self.local_training_data.size(1) - 1:
                        continue
                    data, targets = get_batch(self.local_training_data, i, self.args)
                    self.client_model.zero_grad()
                    optimizer.zero_grad()
                    output = self.client_model.forward(data)

                    # Discard the effective history part
                    eff_history = self.args.seq_len - self.args.validseqlen
                    if eff_history < 0:
                        raise ValueError("Valid sequence length must be smaller than sequence length!")
                    final_target = targets[:, eff_history:].contiguous().view(-1)
                    final_output = output[:, eff_history:].contiguous().view(-1, self.args.n_words)
                    loss = criterion(final_output, final_target)
                    loss.backward()
                    if self.args.max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.client_model.parameters(), self.args.max_norm)
                    optimizer.step()
                    batch_loss.append(loss.item())
            elif self.args.dataset == "shakespeare":
                for batch_idx, (data, targets) in enumerate(self.local_training_data):
                    #data, targets = data.to(self.device), targets.to(self.device)
                    self.client_model.zero_grad()
                    optimizer.zero_grad()
                    output = self.client_model.forward(data)

                    # Discard the effective history part
                    eff_history = data.size(1)-1
                    if eff_history < 0:
                        raise ValueError("Valid sequence length must be smaller than sequence length!")
                    final_target = targets[:, eff_history:].contiguous().view(-1)
                    final_output = output[:, eff_history:].contiguous().view(-1, self.args.n_chars)
                    loss = criterion(final_output, final_target)
                    loss.backward()
                    if self.args.max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.client_model.parameters(), self.args.max_norm)
                    optimizer.step()
                    cur_batch_loss = loss.item()
                    batch_loss.append(cur_batch_loss)
            else:
                for batch_idx, (x, labels) in enumerate(self.local_training_data):
                    x, labels = x.to(self.device), labels.to(self.device)
                    self.client_model.zero_grad()
                    log_probs = self.client_model.forward(x)
                    if self.args.model == "darts":
                        log_probs = log_probs[0]
                    if self.args.kd_ratio > 0:
                        with torch.no_grad():
                            soft_logits = self.teacher_model.forward(x).detach()
                            soft_label = F.softmax(soft_logits, dim=1)
                    if self.args.kd_ratio == 0:
                        loss = criterion(log_probs, labels)
                    else:
                        if self.args.kd_type == "ce":
                            kd_loss = self.cross_entropy_loss_with_soft_target(
                                log_probs, soft_label
                            )
                        else:
                            kd_loss = F.mse_loss(log_probs, soft_logits)
                        loss = self.args.kd_ratio * kd_loss + (
                            1 - self.args.kd_ratio
                        ) * criterion(log_probs, labels)
                    loss.backward()

                    # to avoid nan loss
                    torch.nn.utils.clip_grad_norm_(
                        self.client_model.parameters(), self.args.max_norm
                    )

                    optimizer.step()

                    batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            if self.args.verbose:
                logging.info(
                    "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                        self.client_idx, epoch, sum(epoch_loss) / len(epoch_loss),
                    )
                )

            # reset tree after every epoch to test memory remains in check (it does). No accuracy obtained for this run.
            # if self.args.client_optimizer == "ftrl" and hasattr(optimizer, 'tree_aggregator'):
            #     if optimizer.tree_aggregator:
            #         optimizer.tree_aggregator.tree = []  # Clear the tree
            #         optimizer.tree_aggregator.step = 1   # Reset step count
            #         optimizer.tree_aggregator.depth = 0  # Reset depth if needed
            #         optimizer.tree_aggregator.max_nodes = 1
            #     torch.cuda.empty_cache()
        if not self.args.use_bn:
            for m in self.client_model.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    with torch.no_grad():
                        assert (
                            m.weight.equal(torch.ones_like(m.weight)),
                            "BN weight param not all 1s",
                        )
                        assert (
                            m.bias.equal(torch.zeros_like(m.bias)),
                            "BN bias param not all 0s",
                        )
                        assert (
                            m.running_mean.equal(torch.zeros_like(m.running_mean)),
                            "BN running mean param not all 0s",
                        )
                        assert (
                            m.running_var.equal(torch.ones_like(m.running_var)),
                            "BN running var param not all 1s",
                        )
        return self.client_model

    def test(self, dataset, args, **kwargs):
        model = self.test_model

        model.to(self.device)
        model.eval()
        if not args.use_bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    with torch.no_grad():
                        assert (
                            m.weight.equal(torch.ones_like(m.weight)),
                            "BN weight param not all 1s",
                        )
                        assert (
                            m.bias.equal(torch.zeros_like(m.bias)),
                            "BN bias param not all 0s",
                        )
                        assert (
                            m.running_mean.equal(torch.zeros_like(m.running_mean)),
                            "BN running mean param not all 0s",
                        )
                        assert (
                            m.running_var.equal(torch.ones_like(m.running_var)),
                            "BN running var param not all 1s",
                        )

        criterion = nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            if self.args.dataset == 'ptb':
                metrics = {"test_total": 0, "test_ppl": 0}
                total_loss = 0
                processed_data_size = 0
                for i in range(0, dataset.size(1) - 1, args.validseqlen):
                    if i + args.seq_len - args.validseqlen >= dataset.size(1) - 1:
                        continue
                    data, targets = get_batch(dataset, i, args)
                    output = model.forward(data)

                    # Discard the effective history, just like in training
                    eff_history = args.seq_len - args.validseqlen
                    final_output = output[:, eff_history:].contiguous().view(-1, self.args.n_words)
                    final_target = targets[:, eff_history:].contiguous().view(-1)

                    loss = criterion(final_output, final_target)

                    # Note that we don't add TAR loss here
                    total_loss += (data.size(1) - eff_history) * loss.item()
                    processed_data_size += data.size(1) - eff_history
                metrics["test_loss"] = float(total_loss) / processed_data_size
                metrics["test_ppl"] = np.exp(metrics["test_loss"])
            elif self.args.dataset == "shakespeare":
                metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
                total_loss = 0
                count = 0
                for batch_idx, (data, target) in enumerate(dataset):
                    #data = data.to(self.device)
                    #target = target.to(self.device)
                    output = model.forward(data)

                    # Discard the effective history, just like in training
                    eff_history = data.size(1)-1
                    final_output = output[:, eff_history:].contiguous().view(-1, self.args.n_chars)
                    final_target = target[:, eff_history:].contiguous().view(-1)

                    loss = criterion(final_output, final_target)

                    #need to verify this
                    _, predicted = torch.max(final_output, -1)
                    correct = predicted.eq(final_target).sum()
                    metrics["test_correct"] += correct.item()
                    metrics["test_total"] += final_target.size(0)
                    # Note that we don't add TAR loss here
                    total_loss += loss.data * final_output.size(0)
                    count += final_output.size(0)
                metrics["test_loss"] = float(total_loss.item()) / count * 1.0
            else:
                metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
                for batch_idx, (x, target) in enumerate(dataset):
                    x = x.to(self.device)
                    target = target.to(self.device)
                    pred = model.forward(x)
                    if self.args.model == "darts":
                        pred = pred[0]
                    loss = criterion(pred, target)

                    _, predicted = torch.max(pred, -1)
                    correct = predicted.eq(target).sum()

                    metrics["test_correct"] += correct.item()
                    metrics["test_loss"] += loss.item() * target.size(0)
                    metrics["test_total"] += target.size(0)
        return metrics
