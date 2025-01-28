from abc import ABC, abstractmethod
import numpy as np
import logging
import copy
import wandb
import torch
from ofa.utils import flops_counter as fp
from collections import OrderedDict
import gc
# from fedml_api.dpftrl.dp_ftrl import FTRLM
from fedml_api.dpftrl.dp_ftrl import DPFTRLMServerOptimizer
import sys

"""
Notes:
Local bn skipped
Don't support datasets with following format: self.args.dataset.startswith("stackoverflow"):
REMOVED CERTAIN SKIPS AND IF STATEMENTS IN ADD_SUBNET FUNCTIONS FOR AGGREGATION
PROXIMAL DIST LOSS NOT IMPLEMENTED
LAYERWISE WD NOT IMPLEMENTED
"""


# Note: note filtering non-trainable parameters
def model_vector(model, req_grad=True):
    if not req_grad:
        for p in model.parameters():
            p.requires_grad = False
    param = [p.view(-1) for p in model.parameters()]
    return torch.cat(param, dim=0)


def model_dict_to_vector(model_dict, model_copy, req_grad=True):
    model_copy.load_state_dict(model_dict)
    return model_vector(model_copy, req_grad=req_grad)


class FLOFA_Trainer(ABC):
    def __init__(
        self,
        server_model,
        dataset,
        client_trainer,
        args,
        lr_scheduler,
        wt_avg_sched_method="Uniform",
        teacher_model=None,
        start_round=0,
    ):
        self.server_model = server_model
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.args = args
        self.start_round = start_round
        self.lr_scheduler = lr_scheduler
        self.teacher_model = teacher_model
        self.client_trainer = client_trainer
        self.sampler_args = dict()
        self.server_model.load_client_sample_counts(train_data_local_num_dict)
        self.server_model.set_top_bottom_k(self.args.top_k_maxnet, self.args.bottom_k_maxnet)
        self.sampler_args["client_per_round"] = self.args.client_num_per_round
        self.sampler_args["K"] = self.args.num_multi_archs
        self.sampler_args["diverse_subnets"] = self.args.diverse_subnets
        self.sampler_args["ps_depth_only"] = self.args.ps_depth_only
        self.wt_avg_sched_method = wt_avg_sched_method
        self.weighted_avg_scheduler = dict()
        self.weighted_avg_scheduler["Uniform"] = self.uniform_avg
        self.weighted_avg_scheduler[
            "maxnet_linear_all_subnet"
        ] = self.maxnet_linear_all_subnet
        self.weighted_avg_scheduler[
            "minnet_linear_all_subnet"
        ] = self.minnet_linear_all_subnet
        self.weighted_avg_scheduler[
            "maxnet_cos_all_subnet"
        ] = self.maxnet_cos_all_subnet
        self.weighted_avg_scheduler[
            "minnet_cos_all_subnet"
        ] = self.minnet_cos_all_subnet
        self.weighted_avg_scheduler[
            "maxnet_cos_all_subnet_sandwich"
        ] = self.maxnet_cos_all_subnet_sandwich
        self.weighted_avg_scheduler[
            "phased_maxnet_all_subnet"
        ] = self.phased_maxnet_all_subnet
        self.weighted_avg_scheduler[
            "multikd_phased_maxnet_all_subnet"
        ] = self.multikd_phased_maxnet_all_subnet

        # Wt Avg Scheduler Local Vars
        self.wt_avg_init = False
        self.init_maxnet_avg_wt = None
        self.mid_maxnet_avg_wt = None
        self.final_maxnet_avg_wt = None
        self.init_minnet_avg_wt = None
        self.final_minnet_avg_wt = None
        self.alpha = None
        self.weight_increment = None
        self.weight_increment1 = None
        self.weight_increment2 = None
        self.num_steps = None
        self.num_steps1 = None
        self.num_steps2 = None

        # Best Model Checkpointing
        self.prev_best = 0.0
        self.best_model_interval = self.args.best_model_freq
        
        # DP FTRL Tree Aggregators
        self.client_tree_aggregators = {}
        # NOTE: Doesn't support client optimizers other than SGD
        # Default momentum for SGD is 0 so we set momentum to 0
        server_model_state_dict = self.server_model.get_model_params()
        self.server_ftrl_optimizer = DPFTRLMServerOptimizer(1, 0, args.noise_std, server_model_state_dict)
        self.ftrl_state = self.server_ftrl_optimizer.init_state(server_model_state_dict)
        print("STD DEV", self.ftrl_state.dp_tree_state.value_generator_state)
        # FedDyn alpha and server state
        self.feddyn = self.args.feddyn
        self.feddyn_alpha = args.feddyn_alpha
        self.model_param_len = 0
        if self.feddyn:
            self.model_param_len = model_vector(self.server_model.get_model_copy()).size()[
                0
            ]
        self.server_state = np.zeros(
            (self.args.client_num_in_total, self.model_param_len)
        ).astype("float32")
        self.weighted_feddyn_alpha = np.asarray(
            [
                self.train_data_local_num_dict[client_idx]
                for client_idx in range(self.args.client_num_in_total)
            ]
        )
        self.weighted_feddyn_alpha = (
            (self.weighted_feddyn_alpha / np.sum(self.weighted_feddyn_alpha))
            * self.feddyn_alpha
            * self.args.client_num_in_total
        )

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [
                client_index for client_index in range(client_num_in_total)
            ]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(
                round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(
                range(client_num_in_total), num_clients, replace=False
            )
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _aggregate(self, w_locals):
        w_global_max_net = self.server_model.get_model_params()
        shared_param_count = dict()
        shared_param_sum = dict()
        for key, tensor in w_global_max_net.items():
            shared_param_count[key] = torch.zeros_like(tensor)
            shared_param_sum[key] = torch.zeros_like(tensor)

        # sum the local models.
        for w_local in w_locals:
            self.server_model.add_subnet(shared_param_sum, shared_param_count, w_local)

        # [Aggregate Logic]
        # shared_weights = zero tensors
        # update shared_weights from participating subnets by summing
        # w_supernet_{t+1} = shared_weights * 1/max(number_of_overlaps, 1) + (shared_weights is zero) w_supernet_{t}
        for key in w_global_max_net:
            w_global_max_net[key] = (
                shared_param_sum[key]
                * (
                    1.0
                    / (shared_param_count[key] + (shared_param_count[key] == 0).int())
                )
                + (shared_param_count[key] == 0) * w_global_max_net[key]
            )
        return w_global_max_net

    def train(self):
        if self.args.ckpt_subnets is None:
            self.args.ckpt_subnets = []
            for subnet_id in self.args.diverse_subnets:
                self.args.ckpt_subnets.append(self.args.diverse_subnets[subnet_id])
        for round_idx in range(self.start_round, self.args.comm_round):
            if round_idx >= self.best_model_interval:
                self.best_model_interval += self.args.best_model_freq
                self.prev_best = 0.0
            self.train_one_round(round_idx)
        if self.args.dry_run:
            return
        if self.args.wandb_watch:
            self.server_model.wandb_pass()
        self.server_model.save(
            "finished_checkpoint_data.pt", self.args.ofa_config, self.args.model,
        )

    def train_one_round(self, round_num, local_ep=None, **kwargs):
        client_indexes = self._client_sampling(
            round_num, self.args.client_num_in_total, self.args.client_num_per_round,
        )
        logging.info("client_indexes = " + str(client_indexes))
        self.server_model.set_cli_indices(client_indexes)
        self.server_model.update_sample()
        avg_weights = self.weighted_avg_scheduler[self.wt_avg_sched_method](round_num)
        w_locals = []
        avg_gflops = 0
        avg_params = 0
        training_samples = []

        for idx in range(self.args.client_num_per_round):
            # update dataset
            client_idx = client_indexes[idx]

            self.client_trainer.update_local_dataset(
                client_idx,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )

            client_model_clone = None

            client_model = self.server_model.sample_subnet(
                round_num, client_idx, idx, self.sampler_args
            )

            if self.args.ftrl:
                client_model_clone = copy.deepcopy(client_model.state_dict())

            #DP FTRL
            if self.args.client_optimizer == "ftrl" and client_idx in self.client_tree_aggregators:
                client_model.set_tree_aggregator(self.client_tree_aggregators[client_idx], round_num)
            
            if round_num > 0 and round_num % 3 == 0:
                self.client_tree_aggregators = {}


            if self.args.dry_run:
                flops, _ = fp.profile(client_model.model, (1, 3, 32, 32))
                gflops = flops / 10e8
                avg_gflops += gflops
                params = sum(
                    p.numel()
                    for p in client_model.model.parameters()
                    if p.requires_grad
                )
                avg_params += params
                wandb.log(
                    {f"GFLOPS/idx:{idx}": gflops, "round": round_num,}, step=round_num,
                )
                wandb.log(
                    {f"Parameters/idx:{idx}": params, "round": round_num,},
                    step=round_num,
                )
                print(
                    f"Round: {round_num}, idx: {idx}, subnet arch: {client_model.model_config}, {gflops} gflops and {params} params"
                )
                continue
            client_model.set_avg_wt(avg_weights[idx])
            cur_model_vec = None
            self.client_trainer.set_model(client_model)
            self.client_trainer.set_alpha(self.weighted_feddyn_alpha[client_idx])
            if self.feddyn:
                # send idx^th client' state
                server_model_copy = copy.deepcopy(self.server_model)
                server_model_copy.superimpose_vec(self.server_state[client_idx])
                superimposed_subnet = server_model_copy.get_subnet(
                    **client_model.model_config
                )
                superimposed_subnet_vec = model_vector(
                    superimposed_subnet, req_grad=False
                )
                self.client_trainer.send_client_state(superimposed_subnet_vec)
                server_model_copy = self.server_model.get_model_copy()
                cur_model_vec = model_dict_to_vector(
                    self._aggregate([client_model]), server_model_copy, req_grad=False,
                )
            cur_lr = self.args.lr
            if self.lr_scheduler is not None:
                cur_lr = self.lr_scheduler.get_lr(round_num)
            if self.args.verbose:
                logging.info(
                    f"Client_id: {client_idx} Round Number: {round_num} "
                    f"Subnet: {client_model.model_config} "
                    f" LR: {cur_lr}"
                    f" Counts: {self.server_model.cli_subnet_track[client_idx]}"
                )

            updated_cli_model = self.client_trainer.train(cur_lr, local_ep)
            if self.args.client_optimizer == "ftrl" and client_idx not in self.client_tree_aggregators:
                self.client_tree_aggregators[client_idx] = updated_cli_model.tree_aggregator
            
            training_samples.append(self.client_trainer.get_sample_number())
            # Update client state
            if self.feddyn:
                updated_cli_model_vec = model_dict_to_vector(
                    self._aggregate([updated_cli_model]),
                    self.server_model.get_model_copy(),
                )

                self.server_state[client_idx] += (
                    (updated_cli_model_vec - cur_model_vec).cpu().detach().numpy()
                )
            # move updated client model to cpu
            updated_cli_model.cpu()
            if self.args.ftrl:
                updated_cli_dict = updated_cli_model.state_dict()
                client_delta_dict = OrderedDict()
                for k in updated_cli_dict.keys():
                    # Clip gradient by ftrl_clip
                    client_delta_dict[k] = torch.clamp(updated_cli_dict[k] - client_model_clone[k], -self.args.ftrl_clip, self.args.ftrl_clip)
                updated_cli_model.set_params(client_delta_dict)
            w_locals.append(updated_cli_model)

            
        # Multiply weight with dataset size proportion to global dataset size
        if self.args.weight_dataset:
            total_training_samples = sum(training_samples)
            for idx in range(self.args.client_num_per_round):
                w_locals[idx].set_avg_wt(
                    w_locals[idx].avg_weight
                    * (float(training_samples[idx]) / total_training_samples)
                )
        if self.args.dry_run:
            avg_gflops /= self.args.client_num_per_round
            avg_params /= self.args.client_num_per_round
            wandb.log(
                {f"AVG GFLOPS": avg_gflops, "round": round_num,}, step=round_num,
            )
            wandb.log(
                {f"AVG Parameters": avg_params, "round": round_num,}, step=round_num,
            )
            print(
                f"Round {round_num}\tAVG_GFLOPS:{avg_gflops}\tAVG_Params:{avg_params}"
            )
            return
        supernet_aggregate = self._aggregate(w_locals)
        if self.args.ftrl:
            supernet_aggregate, self.ftrl_state = self.server_ftrl_optimizer.model_update(self.ftrl_state, supernet_aggregate, round_num)

        self.server_model.set_model_params(supernet_aggregate)
        # FedDyn Server Side model aggregation
        if self.feddyn:
            mean_client_states = np.mean(self.server_state, axis=0)
            self.server_model.sum_supernet_w_vec(mean_client_states)

        if self.args.wandb_watch and round_num % self.args.wandb_watch_freq == 0:
            self.server_model.wandb_pass()

        # test results
        # at last round
        if round_num == self.args.comm_round - 1:
            if self.args.efficient_test:
                self._efficient_local_test_on_all_clients(round_num)
            else:
                self._local_test_on_all_clients(round_num)
        # per {frequency_of_the_test} round
        elif round_num % self.args.frequency_of_the_test == 0:
            if self.args.efficient_test:
                (_, subnet_test_acc_map,) = self._efficient_local_test_on_all_clients(
                    round_num
                )
            else:
                _, subnet_test_acc_map = self._local_test_on_all_clients(round_num)
            mean_acc = 0.0
            for subnet_arch in self.args.ckpt_subnets:
                self.client_trainer.set_test_model(
                    self.server_model.get_subnet(**subnet_arch),
                )
                test_metrics = self.client_trainer.local_test(True)
                if self.args.dataset == 'ptb':
                    mean_acc += test_metrics["test_ppl"]
                else:
                    mean_acc += test_metrics["test_correct"] / test_metrics["test_total"]
            mean_acc /= len(self.args.ckpt_subnets)
            if self.args.dataset == 'ptb':
                wandb.log(
                    {f"Test/Mean/PPL": mean_acc, "round": round_num,}, step=round_num,
                )
                if mean_acc < self.prev_best:
                    self.prev_best = mean_acc
                    self.server_model.save(
                        f"best_checkpoint_supernet_{self.best_model_interval}.pt",
                        self.args.ofa_config,
                        self.args.model,
                    )
            else:
                wandb.log(
                    {f"Test/Mean/Acc": mean_acc, "round": round_num,}, step=round_num,
                )
                if mean_acc > self.prev_best:
                    self.prev_best = mean_acc
                    self.server_model.save(
                        f"best_checkpoint_supernet_{self.best_model_interval}.pt",
                        self.args.ofa_config,
                        self.args.model,
                    )
        # Checkpointing
        if round_num > 0 and round_num % self.args.model_checkpoint_freq == 0:
            self.server_model.save(
                f"finished_checkpoint_data_{round_num}.pt",
                self.args.ofa_config,
                self.args.model,
            )
        
        # reset tree aggreggator dict every x rounds (in hopes of curbing memory balloning)
        if round_num > 0 and round_num % 3 == 0:
            print(f"/////////////////////////resetting tree aggregators so we dont get killed, ref_count({sys.getrefcount(self.client_tree_aggregators)})/////////////////////////////")
            for key in list(self.client_tree_aggregators.keys()):
                del self.client_tree_aggregators[key]
            
            self.client_tree_aggregators = {}
            gc.collect()

        
            # torch.cuda.empty_cache()

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        avg_subnet_train_metrics = {
            "num_samples": [],
            "num_correct": [],
            "losses": [],
        }

        avg_subnet_test_metrics = {
            "num_samples": [],
            "num_correct": [],
            "losses": [],
        }
        subnet_test_acc_map = dict()
        subnet_train_acc_map = dict()
        for subnet_id in self.args.diverse_subnets:
            subnet_info = self.args.diverse_subnets[subnet_id]
            train_metrics = {
                "num_samples": [],
                "num_correct": [],
                "losses": [],
            }

            test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

            for client_idx in range(self.args.client_num_in_total):
                """
                Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
                the training client number is larger than the testing client number
                """
                if self.test_data_local_dict[client_idx] is None:
                    continue

                self.client_trainer.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                )

                # set model first
                self.client_trainer.set_test_model(
                    self.server_model.get_subnet(
                        **self.args.diverse_subnets[subnet_id]
                    ),
                )

                # train data
                train_local_metrics = self.client_trainer.local_test(False)
                train_metrics["num_samples"].append(
                    copy.deepcopy(train_local_metrics["test_total"])
                )
                train_metrics["num_correct"].append(
                    copy.deepcopy(train_local_metrics["test_correct"])
                )
                train_metrics["losses"].append(
                    copy.deepcopy(train_local_metrics["test_loss"])
                )
                if self.args.verbose_test:
                    print("train stats", client_idx, train_local_metrics)

                # test data
                test_local_metrics = self.client_trainer.local_test(True)
                test_metrics["num_samples"].append(
                    copy.deepcopy(test_local_metrics["test_total"])
                )
                test_metrics["num_correct"].append(
                    copy.deepcopy(test_local_metrics["test_correct"])
                )
                test_metrics["losses"].append(
                    copy.deepcopy(test_local_metrics["test_loss"])
                )
                if self.args.verbose_test:
                    print("test stats", client_idx, test_local_metrics)

                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break

            # test on training dataset
            train_acc = sum(train_metrics["num_correct"]) / sum(
                train_metrics["num_samples"]
            )
            train_loss = sum(train_metrics["losses"]) / sum(
                train_metrics["num_samples"]
            )
            avg_subnet_train_metrics["num_correct"].append(train_acc)
            avg_subnet_train_metrics["losses"].append(train_loss)
            avg_subnet_train_metrics["num_samples"].append(1)

            # test on test dataset
            test_acc = sum(test_metrics["num_correct"]) / sum(
                test_metrics["num_samples"]
            )
            test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])
            avg_subnet_test_metrics["num_correct"].append(test_acc)
            avg_subnet_test_metrics["losses"].append(test_loss)
            avg_subnet_test_metrics["num_samples"].append(1)

            stats = {
                "training_acc": train_acc,
                "training_loss": train_loss,
                "subnet": subnet_info,
            }

            wandb_subnet_log = dict()
            if "d" in subnet_info:
                wandb_subnet_log["d"] = subnet_info["d"]
            if "e" in subnet_info:
                wandb_subnet_log["e"] = subnet_info["e"]
            if "w" in subnet_info:
                wandb_subnet_log["w"] = subnet_info["w"]

            wandb.log(
                {f"Train/{wandb_subnet_log}/Acc": train_acc, "round": round_idx,},
                step=round_idx,
            )
            wandb.log(
                {f"Train/{wandb_subnet_log}/Loss": train_loss, "round": round_idx,},
                step=round_idx,
            )

            if self.args.verbose:
                logging.info(stats)

            subnet_train_acc_map[subnet_id] = stats["training_acc"]
            stats = {
                "test_acc": test_acc,
                "test_loss": test_loss,
                "subnet": subnet_info,
            }
            wandb.log(
                {f"Test/{wandb_subnet_log}/Acc": test_acc, "round": round_idx},
                step=round_idx,
            )
            wandb.log(
                {f"Test/{wandb_subnet_log}/Loss": test_loss, "round": round_idx,},
                step=round_idx,
            )
            logging.info(stats)
            subnet_test_acc_map[subnet_id] = stats["test_acc"]

        final_train_acc = sum(avg_subnet_train_metrics["num_correct"]) / sum(
            avg_subnet_train_metrics["num_samples"]
        )
        final_train_loss = sum(avg_subnet_train_metrics["losses"]) / sum(
            avg_subnet_train_metrics["num_samples"]
        )
        # test on test dataset
        final_test_acc = sum(avg_subnet_test_metrics["num_correct"]) / sum(
            avg_subnet_test_metrics["num_samples"]
        )
        final_test_loss = sum(avg_subnet_test_metrics["losses"]) / sum(
            avg_subnet_test_metrics["num_samples"]
        )
        final_stats = {
            "final_training_acc": train_acc,
            "final_training_loss": train_loss,
        }
        wandb.log({"Train/Acc": final_train_acc, "round": round_idx}, step=round_idx)
        wandb.log(
            {"Train/Loss": final_train_loss, "round": round_idx}, step=round_idx,
        )
        if self.args.verbose:
            logging.info(final_stats)

        final_stats = {
            "final_test_acc": test_acc,
            "final_test_loss": test_loss,
        }
        wandb.log({f"Test/Acc": final_test_acc, "round": round_idx}, step=round_idx)
        wandb.log({f"Test/Loss": final_test_loss, "round": round_idx}, step=round_idx)
        logging.info(final_stats)
        return subnet_train_acc_map, subnet_test_acc_map

    def _efficient_local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        avg_subnet_train_metrics = {
            "num_samples": [],
            "num_correct": [],
            "ppl": [],
            "losses": [],
        }

        avg_subnet_test_metrics = {
            "num_samples": [],
            "num_correct": [],
            "ppl": [],
            "losses": [],
        }

        subnet_train_acc_map = dict()
        subnet_test_acc_map = dict()

        for subnet_id in self.args.diverse_subnets:
            subnet_info = self.args.diverse_subnets[subnet_id]
            # set model first
            self.client_trainer.set_test_model(
                self.server_model.get_subnet(**subnet_info),
            )
            if not self.args.skip_train_test:
                # gather train partition accuracies
                train_metrics = {"num_samples": [], "num_correct": [], "ppl": [], "losses": []}
                for client_idx in range(self.args.client_num_in_total):
                    self.client_trainer.update_local_dataset(
                        0,
                        self.train_data_local_dict[client_idx],
                        self.test_data_local_dict[0],
                        self.train_data_local_num_dict[client_idx],
                    )
                    # train data
                    train_local_metrics = self.client_trainer.local_test(False)
                    if self.args.dataset == 'ptb':
                        train_metrics["ppl"].append(
                            copy.deepcopy(train_local_metrics["test_ppl"])
                        )
                    else:
                        train_metrics["num_samples"].append(
                            copy.deepcopy(train_local_metrics["test_total"])
                        )
                        train_metrics["num_correct"].append(
                            copy.deepcopy(train_local_metrics["test_correct"])
                        )
                    train_metrics["losses"].append(
                        copy.deepcopy(train_local_metrics["test_loss"])
                    )
                    if self.args.verbose_test:
                        print("train stats", train_local_metrics)
            if self.args.dataset == "shakespeare":
                self.client_trainer.update_local_dataset(
                    0,
                    self.train_data_local_dict[0],
                    self.test_global,
                    self.train_data_local_num_dict[0],
                )
            test_metrics = dict()
            # test data
            test_local_metrics = self.client_trainer.local_test(True)
            if self.args.dataset == 'ptb':
                test_metrics["ppl"] = test_local_metrics["test_ppl"]
            else:
                test_metrics["num_samples"] = test_local_metrics["test_total"]
                test_metrics["num_correct"] = test_local_metrics["test_correct"]
            test_metrics["losses"] = test_local_metrics["test_loss"]

            if self.args.verbose_test:
                print("test stats", test_local_metrics)

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break
            if self.args.dataset == 'ptb':
                if not self.args.skip_train_test:
                    train_loss = sum(train_metrics["losses"]) / self.args.client_num_in_total
                    train_ppl = sum(train_metrics["ppl"]) / self.args.client_num_in_total
                    avg_subnet_train_metrics["ppl"].append(train_ppl)
                    avg_subnet_train_metrics["losses"].append(train_loss)
                    avg_subnet_train_metrics["num_samples"].append(1)

                    wandb_subnet_log = dict()
                    if "d" in subnet_info:
                        wandb_subnet_log["d"] = subnet_info["d"]
                    if "e" in subnet_info:
                        wandb_subnet_log["e"] = subnet_info["e"]

                    stats = {
                        "train_ppl": train_ppl,
                        "train_loss": train_loss,
                        "subnet": subnet_info,
                    }
                    wandb.log(
                        {f"Train/{wandb_subnet_log}/PPL": train_ppl, "round": round_idx,},
                        step=round_idx,
                    )
                    wandb.log(
                        {f"Train/{wandb_subnet_log}/Loss": train_loss, "round": round_idx,},
                        step=round_idx,
                    )
                    logging.info(stats)
                    subnet_train_acc_map[subnet_id] = stats["train_ppl"]

                # test on test dataset
                test_ppl = test_metrics["ppl"]
                test_loss = test_metrics["losses"]
                avg_subnet_test_metrics["ppl"].append(test_ppl)
                avg_subnet_test_metrics["losses"].append(test_loss)
                avg_subnet_test_metrics["num_samples"].append(1)

                wandb_subnet_log = dict()
                if "d" in subnet_info:
                    wandb_subnet_log["d"] = subnet_info["d"]
                if "e" in subnet_info:
                    wandb_subnet_log["e"] = subnet_info["e"]
                if "w" in subnet_info:
                    wandb_subnet_log["w"] = subnet_info["w"]

                stats = {
                    "test_ppl": test_ppl,
                    "test_loss": test_loss,
                    "subnet": subnet_info,
                }
                wandb.log(
                    {f"Test/{wandb_subnet_log}/PPL": test_ppl, "round": round_idx},
                    step=round_idx,
                )
                wandb.log(
                    {f"Test/{wandb_subnet_log}/Loss": test_loss, "round": round_idx,},
                    step=round_idx,
                )
                logging.info(stats)
                subnet_test_acc_map[subnet_id] = stats["test_ppl"]
            else:
                if not self.args.skip_train_test:
                    # test on train dataset
                    train_acc = sum(train_metrics["num_correct"]) / sum(
                        train_metrics["num_samples"]
                    )
                    train_loss = sum(train_metrics["losses"]) / sum(
                        train_metrics["num_samples"]
                    )
                    avg_subnet_train_metrics["num_correct"].append(train_acc)
                    avg_subnet_train_metrics["losses"].append(train_loss)
                    avg_subnet_train_metrics["num_samples"].append(1)

                    wandb_subnet_log = dict()
                    if "d" in subnet_info:
                        wandb_subnet_log["d"] = subnet_info["d"]
                    if "e" in subnet_info:
                        wandb_subnet_log["e"] = subnet_info["e"]
                    if "w" in subnet_info:
                        wandb_subnet_log["w"] = subnet_info["w"]

                    stats = {
                        "train_acc": train_acc,
                        "train_loss": train_loss,
                        "subnet": subnet_info,
                    }
                    wandb.log(
                        {f"Train/{wandb_subnet_log}/Acc": train_acc, "round": round_idx,},
                        step=round_idx,
                    )
                    wandb.log(
                        {f"Train/{wandb_subnet_log}/Loss": train_loss, "round": round_idx,},
                        step=round_idx,
                    )
                    logging.info(stats)
                    subnet_train_acc_map[subnet_id] = stats["train_acc"]

                # test on test dataset
                test_acc = test_metrics["num_correct"] / test_metrics["num_samples"]
                test_loss = test_metrics["losses"] / test_metrics["num_samples"]
                avg_subnet_test_metrics["num_correct"].append(test_acc)
                avg_subnet_test_metrics["losses"].append(test_loss)
                avg_subnet_test_metrics["num_samples"].append(1)

                wandb_subnet_log = dict()
                if "d" in subnet_info:
                    wandb_subnet_log["d"] = subnet_info["d"]
                if "e" in subnet_info:
                    wandb_subnet_log["e"] = subnet_info["e"]
                if "w" in subnet_info:
                    wandb_subnet_log["w"] = subnet_info["w"]

                stats = {
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                    "subnet": subnet_info,
                }
                wandb.log(
                    {f"Test/{wandb_subnet_log}/Acc": test_acc, "round": round_idx},
                    step=round_idx,
                )
                wandb.log(
                    {f"Test/{wandb_subnet_log}/Loss": test_loss, "round": round_idx,},
                    step=round_idx,
                )
                logging.info(stats)
                subnet_test_acc_map[subnet_id] = stats["test_acc"]

        if self.args.dataset == 'ptb':
            if not self.args.skip_train_test:
                # test on train dataset

                final_train_ppl = sum(avg_subnet_train_metrics["ppl"]) / sum(
                    avg_subnet_train_metrics["num_samples"]
                )
                final_train_loss = sum(avg_subnet_train_metrics["losses"]) / sum(
                    avg_subnet_train_metrics["num_samples"]
                )

                final_stats = {
                    "final_train_ppl": final_train_ppl,
                    "final_train_loss": final_train_loss,
                }
                wandb.log({f"Train/PPL": final_train_ppl, "round": round_idx}, step=round_idx)
                wandb.log({f"Train/Loss": final_train_loss, "round": round_idx}, step=round_idx)
                logging.info(final_stats)

            # test on test dataset
            final_test_ppl = sum(avg_subnet_test_metrics["ppl"]) / sum(
                avg_subnet_test_metrics["num_samples"]
            )
            final_test_loss = sum(avg_subnet_test_metrics["losses"]) / sum(
                avg_subnet_test_metrics["num_samples"]
            )

            final_stats = {
                "final_test_ppl": final_test_ppl,
                "final_test_loss": final_test_loss,
            }
            wandb.log({f"Test/PPL": final_test_ppl, "round": round_idx}, step=round_idx)
            wandb.log({f"Test/Loss": final_test_loss, "round": round_idx}, step=round_idx)
            logging.info(final_stats)
        else:
            if not self.args.skip_train_test:
                # test on train dataset
                final_train_acc = sum(avg_subnet_train_metrics["num_correct"]) / sum(
                    avg_subnet_train_metrics["num_samples"]
                )
                final_train_loss = sum(avg_subnet_train_metrics["losses"]) / sum(
                    avg_subnet_train_metrics["num_samples"]
                )

                final_stats = {
                    "final_train_acc": final_train_acc,
                    "final_train_loss": final_train_loss,
                }
                wandb.log({f"Train/Acc": final_train_acc, "round": round_idx}, step=round_idx)
                wandb.log({f"Train/Loss": final_train_loss, "round": round_idx}, step=round_idx)
                logging.info(final_stats)

            # test on test dataset
            final_test_acc = sum(avg_subnet_test_metrics["num_correct"]) / sum(
                avg_subnet_test_metrics["num_samples"]
            )
            final_test_loss = sum(avg_subnet_test_metrics["losses"]) / sum(
                avg_subnet_test_metrics["num_samples"]
            )

            final_stats = {
                "final_test_acc": final_test_acc,
                "final_test_loss": final_test_loss,
            }
            wandb.log({f"Test/Acc": final_test_acc, "round": round_idx}, step=round_idx)
            wandb.log({f"Test/Loss": final_test_loss, "round": round_idx}, step=round_idx)
            logging.info(final_stats)
        return subnet_train_acc_map, subnet_test_acc_map

    def uniform_avg(self, round_num=None):
        subnet_flofa_avg_weights = []
        for i in range(self.args.client_num_per_round):
            subnet_flofa_avg_weights.append(1)
        return subnet_flofa_avg_weights

    def maxnet_linear_all_subnet(self, round_num):
        if not self.wt_avg_init:
            self.init_maxnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.final_maxnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.num_steps = self.args.weighted_avg_schedule["num_steps"]
            self.weight_increment = (
                self.final_maxnet_avg_wt - self.init_maxnet_avg_wt
            ) / self.num_steps
            self.wt_avg_init = True
        round_num = min(round_num, self.num_steps)
        maxnet_weight = self.init_maxnet_avg_wt + self.weight_increment * round_num
        other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
        maxnet_weight /= self.args.top_k_maxnet
        subnet_flofa_avg_weights = []
        for i in range(self.args.client_num_per_round):
            subnet_flofa_avg_weights.append(other_weight)
        for idx in range(self.args.client_num_per_round):
            if (
                self.server_model.cli_indices[idx]
                in self.server_model.largest_subnet_min_idx
            ):
                subnet_flofa_avg_weights[idx] = maxnet_weight
                break
        return subnet_flofa_avg_weights

    def minnet_linear_all_subnet(self, round_num):
        if not self.wt_avg_init:
            self.init_minnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.final_minnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.num_steps = self.args.weighted_avg_schedule["num_steps"]
            self.weight_increment = (
                self.final_minnet_avg_wt - self.init_minnet_avg_wt
            ) / self.num_steps
            self.wt_avg_init = True
        round_num = min(round_num, self.num_steps)
        minnet_weight = self.init_minnet_avg_wt + self.weight_increment * round_num
        other_weight = (1 - minnet_weight) / (self.args.client_num_per_round - self.args.bottom_k_maxnet)
        minnet_weight /= self.args.bottom_k_maxnet
        subnet_flofa_avg_weights = []
        for i in range(self.args.client_num_per_round):
            subnet_flofa_avg_weights.append(other_weight)
        for idx in range(self.args.client_num_per_round):
            if (
                self.server_model.cli_indices[idx]
                in self.server_model.smallest_subnet_min_idx
            ):
                subnet_flofa_avg_weights[idx] = minnet_weight
                break
        return subnet_flofa_avg_weights

    def maxnet_cos_all_subnet(self, round_num):
        if not self.wt_avg_init:
            self.init_maxnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.final_maxnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.alpha = float(self.final_maxnet_avg_wt) / self.init_maxnet_avg_wt
            self.num_steps = self.args.weighted_avg_schedule["num_steps"]
            self.wt_avg_init = True

        round_num = min(round_num, self.num_steps)
        cos_decay = 0.5 * (1 + np.cos(np.pi * round_num / self.num_steps))
        decayed = (1 - self.alpha) * cos_decay + self.alpha
        maxnet_weight = self.init_maxnet_avg_wt * decayed
        other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
        maxnet_weight /= self.args.top_k_maxnet
        subnet_flofa_avg_weights = []
        for i in range(self.args.client_num_per_round):
            subnet_flofa_avg_weights.append(other_weight)
        for idx in range(self.args.client_num_per_round):
            if (
                self.server_model.cli_indices[idx]
                in self.server_model.largest_subnet_min_idx
            ):
                subnet_flofa_avg_weights[idx] = maxnet_weight
                break
        return subnet_flofa_avg_weights

    def maxnet_cos_all_subnet_sandwich(self, round_num):
        if not self.wt_avg_init:
            self.init_maxnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.final_maxnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.alpha = float(self.final_maxnet_avg_wt) / self.init_maxnet_avg_wt
            self.num_steps = self.args.weighted_avg_schedule["num_steps"]
            self.wt_avg_init = True

        round_num = min(round_num, self.num_steps)
        cos_decay = 0.5 * (1 + np.cos(np.pi * round_num / self.num_steps))
        decayed = (1 - self.alpha) * cos_decay + self.alpha
        maxnet_weight = self.init_maxnet_avg_wt * decayed
        other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
        maxnet_weight /= self.args.top_k_maxnet
        subnet_flofa_avg_weights = []
        for i in range(self.args.client_num_per_round):
            subnet_flofa_avg_weights.append(other_weight)
        for idx in range(self.args.client_num_per_round):
            if idx == ((round_num + 1) % self.args.client_num_per_round):
                subnet_flofa_avg_weights[idx] = maxnet_weight
                break
        return subnet_flofa_avg_weights

    def minnet_cos_all_subnet(self, round_num):
        if not self.wt_avg_init:
            self.init_minnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.final_minnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.alpha = float(self.final_minnet_avg_wt) / self.init_minnet_avg_wt
            self.num_steps = self.args.weighted_avg_schedule["num_steps"]
            self.wt_avg_init = True

        round_num = min(round_num, self.num_steps)
        cos_decay = 0.5 * (1 + np.cos(np.pi * round_num / self.num_steps))
        decayed = (1 - self.alpha) * cos_decay + self.alpha
        minnet_weight = self.init_minnet_avg_wt * decayed
        other_weight = (1 - minnet_weight) / (self.args.client_num_per_round - self.args.bottom_k_maxnet)
        minnet_weight /= self.args.bottom_k_maxnet
        subnet_flofa_avg_weights = []
        for i in range(self.args.client_num_per_round):
            subnet_flofa_avg_weights.append(other_weight)
        for idx in range(self.args.client_num_per_round):
            if (
                self.server_model.cli_indices[idx]
                in self.server_model.smallest_subnet_min_idx
            ):
                subnet_flofa_avg_weights[idx] = minnet_weight
                break
        return subnet_flofa_avg_weights

    def phased_maxnet_all_subnet(self, round_num):
        if not self.wt_avg_init:
            self.init_maxnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.mid_maxnet_avg_wt = self.args.weighted_avg_schedule["mid"]
            self.final_maxnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.num_steps1 = self.args.weighted_avg_schedule["num_steps1"]
            self.num_steps2 = self.args.weighted_avg_schedule["num_steps2"]
            self.alpha = float(self.final_maxnet_avg_wt) / self.mid_maxnet_avg_wt
            self.weight_increment1 = (
                self.mid_maxnet_avg_wt - self.init_maxnet_avg_wt
            ) / self.num_steps1
            self.weight_increment2 = (
                self.final_maxnet_avg_wt - self.mid_maxnet_avg_wt
            ) / self.num_steps2
            self.wt_avg_init = True
        if round_num < self.num_steps1:
            maxnet_weight = self.init_maxnet_avg_wt + self.weight_increment1 * round_num
            other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
            maxnet_weight /= self.args.top_k_maxnet
            subnet_flofa_avg_weights = []
            for i in range(self.args.client_num_per_round):
                subnet_flofa_avg_weights.append(other_weight)
            for idx in range(self.args.client_num_per_round):
                if (
                    self.server_model.cli_indices[idx]
                    in self.server_model.largest_subnet_min_idx
                ):
                    subnet_flofa_avg_weights[idx] = maxnet_weight
                    break
        else:
            round_num = min(round_num, self.num_steps1 + self.num_steps2)
            cos_decay = 0.5 * (
                1 + np.cos(np.pi * round_num / self.num_steps1 + self.num_steps2)
            )
            decayed = (1 - self.alpha) * cos_decay + self.alpha
            maxnet_weight = self.init_maxnet_avg_wt * decayed
            other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
            maxnet_weight /= self.args.top_k_maxnet
            subnet_flofa_avg_weights = []
            for i in range(self.args.client_num_per_round):
                subnet_flofa_avg_weights.append(other_weight)
            for idx in range(self.args.client_num_per_round):
                if (
                    self.server_model.cli_indices[idx]
                    in self.server_model.largest_subnet_min_idx
                ):
                    subnet_flofa_avg_weights[idx] = maxnet_weight
                    break
        return subnet_flofa_avg_weights

    def multikd_phased_maxnet_all_subnet(self, round_num):
        if not self.wt_avg_init:
            self.init_maxnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.mid_maxnet_avg_wt = self.args.weighted_avg_schedule["mid"]
            self.final_maxnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.num_steps1 = self.args.weighted_avg_schedule["num_steps1"]
            self.num_steps2 = self.args.weighted_avg_schedule["num_steps2"]
            self.alpha = float(self.final_maxnet_avg_wt) / self.mid_maxnet_avg_wt
            self.weight_increment1 = (
                self.mid_maxnet_avg_wt - self.init_maxnet_avg_wt
            ) / self.num_steps1
            self.weight_increment2 = (
                self.final_maxnet_avg_wt - self.mid_maxnet_avg_wt
            ) / self.num_steps2
            self.wt_avg_init = True
        if round_num < self.num_steps1:
            maxnet_weight = self.init_maxnet_avg_wt + self.weight_increment1 * round_num
            other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
            maxnet_weight /= self.args.top_k_maxnet
            subnet_flofa_avg_weights = []
            for i in range(self.args.client_num_per_round):
                subnet_flofa_avg_weights.append(other_weight)
                subnet_flofa_avg_weights.append(maxnet_weight)
            for idx in range(self.args.client_num_per_round):
                if (
                    self.server_model.cli_indices[idx]
                    in self.server_model.largest_subnet_min_idx
                ):
                    subnet_flofa_avg_weights.pop(idx * 2)
                    break
        else:
            round_num = min(round_num, self.num_steps1 + self.num_steps2)
            cos_decay = 0.5 * (
                1 + np.cos(np.pi * round_num / self.num_steps1 + self.num_steps2)
            )
            decayed = (1 - self.alpha) * cos_decay + self.alpha
            maxnet_weight = self.init_maxnet_avg_wt * decayed
            other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
            maxnet_weight /= self.args.top_k_maxnet
            subnet_flofa_avg_weights = []
            for i in range(self.args.client_num_per_round):
                subnet_flofa_avg_weights.append(other_weight)
                subnet_flofa_avg_weights.append(maxnet_weight)
            for idx in range(self.args.client_num_per_round):
                if (
                    self.server_model.cli_indices[idx]
                    in self.server_model.largest_subnet_min_idx
                ):
                    subnet_flofa_avg_weights.pop(idx * 2)
                    break
        return subnet_flofa_avg_weights

    # FedDyn update server state variables
    def update_server_state(self, client_updates):
        model_delta = self.server_model.get_model_copy()
        for param in model_delta.parameters():
            param.data = torch.zeros_like(param.data)
        global_model = self.server_model.get_model_copy()
        for client_model in client_updates:
            for server_param, client_param, delta_param in zip(
                global_model.parameters(),
                client_model.parameters(),
                model_delta.parameters(),
            ):
                delta_param.data += (
                    client_param - server_param
                ) / self.args.client_num_in_total

        for state_param, delta_param in zip(
            self.server_model.get_server_state().parameters(), model_delta.parameters(),
        ):
            state_param.data -= self.feddyn_alpha * delta_param
