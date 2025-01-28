import gc

class ClientModel:
    def __init__(
        self,
        model,
        model_index,
        model_config,
        is_max_net,
        sample_random_subnet=None,
        sample_random_depth_subnet=None,
        tree_aggregator = None,
        avg_weight=1,
    ):
        self.model = model
        self.model_index = model_index
        self.model_config = model_config
        self.is_max_net = is_max_net
        self.avg_weight = avg_weight
        self.sample_random_subnet = sample_random_subnet
        self.sample_random_depth_subnet = sample_random_depth_subnet
        self.tree_aggregator = tree_aggregator
        
    def get_model(self):
        return self.model

    def state_dict(self):
        return self.model.cpu().state_dict()

    def set_params(self, state_dict):
        self.model.load_state_dict(state_dict)

    def to(self, device):
        self.model.to(device)

    def cpu(self):
        self.model = self.model.cpu()

    def set_tree_aggregator(self, tree_aggregator, round_num):
        # if round_num > 0 and round_num % 3 == 0:
        #     print("//////Trying to avoid OOM//////")
        #     self.tree_aggregator = None
        #     gc.collect()
        self.tree_aggregator = tree_aggregator
    
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def modules(self):
        return self.model.modules()

    def parameters(self):
        return self.model.parameters()

    def zero_grad(self):
        self.model.zero_grad()

    def forward(self, x):
        return self.model(x)

    def set_avg_wt(self, a):
        self.avg_weight = a

    def freeze(self):
        for params in self.model.parameters():
            params.requires_grad = False
        self.model.eval()

    def set_active_subnet(self, arch):
        self.model.set_active_subnet(**arch)

    def set_max_net(self):
        self.model.set_max_net()
