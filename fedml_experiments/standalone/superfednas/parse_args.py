import json

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument(
        "--model",
        type=str,
        default="resnet56",
        metavar="N",
        help="neural network used in training",
    )

    parser.add_argument(
        "--multi",
        action="store_true",
        help="Pass on multiple networks to a single client",
    )

    parser.add_argument(
        "--skip_train_largest",
        action="store_true",
        help="Pass on multiple networks to a single client",
    )

    parser.add_argument(
        "--multi_drop_largest",
        action="store_true",
        help="(Needs to be reworked!) drops largest model and smallest to allow single middle model to be aggregated only",
    )

    parser.add_argument(
        "--cli_supernet",
        action="store_true",
        help="Pass on entire supernetwork to a single client",
    )

    parser.add_argument(
        "--cli_supernet_ps",
        action="store_true",
        help="Pass on entire supernetwork to a single client and perform PS on each client",
    )

    parser.add_argument(
        "--inplace_kd", action="store_true", help="In place knowledge distillation",
    )

    parser.add_argument(
        "--multi_disable_rest_bn",
        action="store_true",
        help="Multi setting Disable BN for rest of the networks apart from the largest one",
    )

    parser.add_argument(
        "--num_multi_archs",
        type=int,
        default=1,
        metavar="K",
        help="Number of middle architectures for multi net run",
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="superfednas",
        metavar="N",
        help="Project Name of wandb for logging",
    )

    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="superfednas",
        metavar="N",
        help="Run Name of wandb for logging",
    )

    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="superfednas",
        metavar="N",
        help="helps in creating a project under teams or personal username",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        metavar="N",
        help="dataset used for training",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./../../../data/cifar10",
        help="data directory",
    )

    parser.add_argument(
        "--partition_method",
        type=str,
        default="homo",
        metavar="N",
        help="how to partition the dataset on local workers",
    )

    parser.add_argument(
        "--partition_alpha",
        type=float,
        default=1,
        metavar="PA",
        help="partition alpha (default: 1)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--client_optimizer", type=str, default="sgd", help="SGD with momentum; adam",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.1)",
    )

    parser.add_argument(
        "--warmup_init_lr",
        type=float,
        default=0,
        help="warmup learning rate init (default: 0)",
    )

    parser.add_argument(
        "--warmup_rounds",
        type=int,
        default=0,
        help="Number of rounds to perform LR warmup",
    )

    parser.add_argument("--wd", help="weight decay parameter;", type=float, default=0)

    parser.add_argument(
        "--largest_subnet_wd",
        help="weight decay parameter for largest subnet",
        type=float,
        default=0,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="EP",
        help="how many epochs will be trained locally",
    )

    parser.add_argument(
        "--client_num_in_total",
        type=int,
        default=20,
        metavar="NN",
        help="number of workers in a distributed cluster",
    )

    parser.add_argument(
        "--client_num_per_round",
        type=int,
        default=8,
        metavar="NN",
        help="number of workers",
    )

    parser.add_argument(
        "--comm_round",
        type=int,
        default=1000,
        help="how many round of communications we shoud use",
    )

    parser.add_argument(
        "--frequency_of_the_test",
        type=int,
        default=50,
        help="the frequency of the algorithms",
    )

    parser.add_argument("--gpu", type=int, default=0, help="gpu")

    parser.add_argument("--ci", type=int, default=0, help="CI")

    parser.add_argument(
        "--diverse_subnets",
        type=json.loads,
        default='{"0":{"d":[2, 2, 2, 2, 2],"w0_avg":1,"wf_avg":1}}',
        help="Subnets participating in training and possibly their starting and final weights during averaging",
    )

    parser.add_argument(
        "--ckpt_subnets",
        type=json.loads,
        default=None,
        help="List of subnets used to checkpoint best model on interval",
    )

    parser.add_argument(
        "--use_bn", action="store_true", help="Use batchnorm",
    )

    parser.add_argument(
        "--bn_gamma_zero_init",
        action="store_true",
        default=False,
        help="Sets the learnable scaling coefficient γ = 0 in the last Batch Normalization layer of each residual block",
    )

    parser.add_argument(
        "--reset_bn_stats",
        action="store_true",
        help="Resets bn mean and variance before test and training",
        default=False,
    )

    parser.add_argument(
        "--reset_bn_stats_test",
        action="store_true",
        help="Resets bn mean and variance before testing",
    )

    parser.add_argument(
        "--efficient_test", action="store_true", help="Test efficiently",
    )

    parser.add_argument(
        "--reset_bn_sample_size",
        type=float,
        default=0.2,
        help="percentage of local train data to use to reset bn stats",
    )

    parser.add_argument(
        "--subnet_dist_type",
        type=str,
        default="all_random",
        choices=[
            "static",
            "dynamic",
            "all_random",
            "compound",
            "sandwich_all_random",
            "sandwich_compound",
            "TS_all_random",
            "TS_compound",
            "max_sample_count",
            "multi_sandwich",
            "TS_KD",
            "PS",
        ],
        help="Subnetwork selection strategy for every round",
    )

    parser.add_argument(
        "--weighted_avg_schedule",
        type=json.loads,
        default=None,
        help="Dynamic Weighted Average Hyper-parameters",
    )

    parser.add_argument(
        "--model_ckpt_name", type=str, default=None, help="Name of model to load",
    )

    parser.add_argument(
        "--run_path", type=str, default=None, help="Run path of model to load",
    )

    parser.add_argument(
        "--cli_subnet_track",
        type=json.loads,
        default=None,
        help="Subnet tracker state for resuming run",
    )

    parser.add_argument(
        "--teacher_ckpt_name",
        type=str,
        default=None,
        help="Name of teacher model to load",
    )

    parser.add_argument(
        "--teacher_run_path",
        type=str,
        default=None,
        help="Run path of teacher model to load",
    )

    parser.add_argument(
        "--ofa_config", type=json.loads, default=None, help="OFA ResNet configuration",
    )

    parser.add_argument(
        "--kd_ratio",
        type=float,
        default=0,
        help="knowledge Distillation using pretrained model",
    )

    parser.add_argument(
        "--kd_type",
        type=str,
        default="ce",
        choices=["ce", "mse"],
        help="knowledge Distillation using pretrained model",
    )

    parser.add_argument(
        "--lr_schedule", type=json.loads, default=None, help="Weighted random"
    )

    parser.add_argument(
        "--wandb_watch", action="store_true", help="Watch parameters on wandb",
    )

    parser.add_argument(
        "--wandb_watch_freq", type=int, default=100, help="wandb watch frequency",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Just simulate distribution of subnetworks and log number of parameters and gflops of each subnet",
    )

    parser.add_argument(
        "--model_checkpoint_freq",
        type=int,
        default=500,
        help="frequency at which model is checkpointed",
    )

    parser.add_argument(
        "--best_model_freq",
        type=int,
        default=1000,
        help="intervals at which to track best model",
    )

    parser.add_argument(
        "--resume_round", type=int, default=-1, help="round to resume training at",
    )

    parser.add_argument(
        "--custom_config",
        type=json.loads,
        default=None,
        help="Allows user to specify custom config.yaml for crashed runs or runs missing config.yaml",
    )

    parser.add_argument(
        "--optim_step_more",
        action="store_true",
        help="Take optimizer step after each gradient in cli supernet",
    )

    parser.add_argument(
        "--largest_step_more",
        action="store_true",
        help="Take optimizer step after largest subnet gradient in cli supernet",
    )

    parser.add_argument(
        "--init_seed",
        type=int,
        default=0,
        help="Initial seed for intializing model weights and partitioning dataset amongst clients",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Log in more detail",
    )

    parser.add_argument(
        "--verbose_test", action="store_true", help="Log testing in more detail",
    )

    parser.add_argument(
        "--feddyn", action="store_true", help="Use FedDyn Optimization",
    )

    parser.add_argument(
        "--feddyn_alpha",
        type=float,
        default=0.01,
        help="FedDyn optimization hyperparameter alpha. Generally one of [0.1, 0.01, 0.001]",
    )
    parser.add_argument(
        "--max_norm",
        type=float,
        default=1.0,
        help="optimization gradient clipping max_norm",
    )
    parser.add_argument(
        "--feddyn_max_norm",
        type=float,
        default=10,
        help="FedDyn optimization gradient clipping max_norm",
    )
    parser.add_argument(
        "--feddyn_no_wd_modifier",
        action="store_true",
        help="Don't use feddyn modified wd",
    )

    parser.add_argument(
        "--feddyn_override_wd",
        type=float,
        default=-1,
        help="Override feddyn wd if > 0",
    )

    parser.add_argument(
        "--mod_wd_dyn",
        action="store_true",
        help="Use FedDyn modified wd in non-feddyn runs",
    )

    parser.add_argument(
        "--weight_dataset",
        action="store_true",
        help="Include dataset size into weighted averaging",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.45,
        help="Elastic TCN dropout",
    )

    parser.add_argument(
        "--emb_dropout",
        type=float,
        default=0.25,
        help="Elastic TCN embedding dropout",
    )

    parser.add_argument(
        "--ksize",
        type=int,
        default=3,
        help="Elastic TCN kernel size",
    )

    parser.add_argument(
        "--emsize",
        type=int,
        default=600,
        help="Elastic TCN size of word embeddings (default: 600)",
    )

    parser.add_argument(
        "--levels",
        type=int,
        default=4,
        help="Elastic TCN # of levels (default: 4)",
    )

    parser.add_argument(
        "--nhid",
        type=int,
        default=600,
        help="Elastic TCN number of hidden units per layer (default: 600)",
    )

    parser.add_argument(
        "--tied",
        action="store_false",
        help="Include dataset tie the encoder-decoder weights (default: True)",
    )

    parser.add_argument(
        "--validseqlen",
        type=int,
        default=40,
        help="Elastic TCN valid sequence length (default: 40)",
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=80,
        help="Elastic TCN total sequence length, including effective history (default: 80)",
    )

    parser.add_argument(
        "--skip_train_test",
        action="store_true",
        help="skip train ppl calculations",
    )

    parser.add_argument(
        "--top_k_maxnet",
        type=int,
        default=1,
        help="How many maximum subnets to sample per round. Must have top_k_maxnet+bottom_k_maxnet <= num clients per round",
    )

    parser.add_argument(
        "--bottom_k_maxnet",
        type=int,
        default=1,
        help="How many minimum subnets to sample per round. Must have top_k_maxnet+bottom_k_maxnet <= num clients per round",
    )
    parser.add_argument(
        "--init_channel_size",
        type=int,
        default=128,
        help="Initial number of channels in convolution of darts model",
    )
    parser.add_argument(
        "--darts_layers",
        type=int,
        default=20,
        help="Number of layers in darts model",
    )
    parser.add_argument(
        "--ps_depth_only",
        type=int,
        default=750,
        help="Number of rounds to only sample depth elastic random models",
    )

    parser.add_argument(
        "--PS_with_largest",
        action="store_true",
        help="During client supernet training using PS, select the largest subnetwork once and sample k-1 random subnets based on phase",
    )

    parser.add_argument(
        "--use_train_pkl", action="store_true", help="Use train dataset generated after splitting original train dataset into train and val datasets",
    )

    parser.add_argument(
        "--noise_multiplier",
        type=float,
        default=7.0,
        help="Noise multiplier for dp ftrl",
    )

    parser.add_argument(
        "--noise_std",
        type=float,
        default=1.0,
        help="Noise std for dp ftrl",
    )

    parser.add_argument(
        "--ftrl_clip",
        type=float,
        default=2.0,
        help="Gradient clipping norm for dp ftrl",
    )

    parser.add_argument(
        "--ftrl", action="store_true", help="Use DP FTRL Server Optimizer",
    )

    return parser

    

def add_args_test(parser):
    parser.add_argument(
        "--model",
        type=str,
        default="resnet56",
        metavar="N",
        help="neural network used in training",
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="superfednas",
        metavar="N",
        help="Project Name of wandb for logging",
    )

    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="superfednas",
        metavar="N",
        help="Run Name of wandb for logging",
    )

    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="superfednas",
        metavar="N",
        help="helps in creating a project under teams or personal username",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        metavar="N",
        help="dataset used for training",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./../../../data/cifar10",
        help="data directory",
    )

    parser.add_argument(
        "--partition_method",
        type=str,
        default="homo",
        metavar="N",
        help="how to partition the dataset on local workers",
    )

    parser.add_argument(
        "--partition_alpha",
        type=float,
        default=1,
        metavar="PA",
        help="partition alpha (default: 1)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--client_num_in_total",
        type=int,
        default=20,
        metavar="NN",
        help="number of workers in a distributed cluster",
    )

    parser.add_argument("--gpu", type=int, default=0, help="gpu")

    parser.add_argument(
        "--model_ckpt_name", type=str, default=None, help="Name of model to load",
    )

    parser.add_argument(
        "--run_path", type=str, default=None, help="Run path of model to load",
    )

    parser.add_argument(
        "--init_seed",
        type=int,
        default=0,
        help="Initial seed for intializing model weights and partitioning dataset amongst clients",
    )

    parser.add_argument(
        "--test_subnets",
        type=json.loads,
        default=None,
        help="Subnets to test as a list",
    )

    parser.add_argument(
        "--nas", action="store_true", help="Perform NAS",
    )

    parser.add_argument(
        "--nas_constraints",
        type=json.loads,
        default=None,
        help="List of constraints to be used during nas",
    )

    parser.add_argument(
        "--mutate_prob", type=float, default=0.1, help="nas mutation probability",
    )

    parser.add_argument(
        "--population_size", type=int, default=100, help="nas population size",
    )

    parser.add_argument(
        "--max_time_budget", type=int, default=500, help="nas max time taken to sample",
    )

    parser.add_argument(
        "--parent_ratio", type=float, default=0.25, help="nas parent ratio",
    )

    parser.add_argument(
        "--mutation_ratio", type=float, default=0.5, help="nas mutation ratio",
    )

    parser.add_argument(
        "--multi_seed_test",
        action="store_true",
        help="Perform test on multiple seed and save results for plotting",
    )

    parser.add_argument(
        "--seed_list",
        type=json.loads,
        default=None,
        help="List of init seed to be used during multi seed test",
    )

    parser.add_argument(
        "--multi_seed_run_paths",
        type=json.loads,
        default=None,
        help="run path for model to test as a list corresponding to seed_list",
    )

    parser.add_argument(
        "--multi_seed_model_ckpt_names",
        type=json.loads,
        default=None,
        help="names of model to load from run paths",
    )

    parser.add_argument(
        "--multi_seed_test_subnets",
        type=json.loads,
        default=None,
        help="Subnets to test as a list for multi seed testing",
    )

    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="path and name of csv to save test data",
    )

    parser.add_argument(
        "--feddyn", action="store_true", help="Use FedDyn Optimization",
    )

    parser.add_argument(
        "--feddyn_alpha",
        type=float,
        default=0.1,
        help="FedDyn optimization hyperparameter alpha. Generally one of [0.1, 0.01, 0.001]",
    )

    return parser
