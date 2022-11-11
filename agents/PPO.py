import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.distributions import Normal, Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Optional, Union
import pdb
import sys
import gin
import os

# To set an initialization similar to TF2
# https://discuss.pytorch.org/t/how-i-can-set-an-initialization-for-conv-kernels-similarly-to-keras/30473
# lr scheduling
# https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling
################################ Class to create a Deep Q Network model ################################
class PPOActorCritic(nn.Module):
    def __init__(
        self,
        seed: int,
        input_shape: int,
        activation: str,
        hidden_units_value: list,
        hidden_units_actor: list,
        num_actions: int,
        batch_norm_input: bool,
        batch_norm_value_out: bool,
        policy_type: str,
        init_std: float = 0.5,
        min_std: float = 0.003,
        std_transform: str = "softplus",
        init_last_layers: str = "rescaled",
        modelname: str = "PPO",
    ):

        super(PPOActorCritic, self).__init__()
        self.seed = seed
        torch.manual_seed(seed)
        self.modelname = modelname
        # set dimensionality of input/output depending on the model
        inp_dim = input_shape[0]
        out_dim = num_actions
        self.policy_type = policy_type
        self.std_transform = std_transform
        self.init_last_layers = init_last_layers
        
        # set flag for batch norm as attribute
        self.bnflag_input = batch_norm_input

        critic_modules = []

        if self.bnflag_input:
            # affine false sould be equal to center and scale to False in TF2
            critic_modules.append(
                nn.BatchNorm1d(inp_dim, affine=False)
            )  # , momentum=1.0
            # critic_modules.append(nn.LayerNorm(inp_dim, elementwise_affine=False))

        # self.input_layer = InputLayer(input_shape=inp_shape)
        # set of hidden layers
        for i in range(len(hidden_units_value)):
            if i == 0:
                critic_modules.append(nn.Linear(inp_dim, hidden_units_value[i]))
                if activation == "relu":
                    critic_modules.append(nn.ReLU())
                elif activation == "tanh":
                    critic_modules.append(nn.Tanh())
                elif activation == "linear":
                    pass
                else:
                    print("Activation selected not available")
                    sys.exit()
            else:
                critic_modules.append(
                    nn.Linear(hidden_units_value[i - 1], hidden_units_value[i])
                )
                if activation == "relu":
                    critic_modules.append(nn.ReLU())
                elif activation == "tanh":
                    critic_modules.append(nn.Tanh())
                elif activation == "linear":
                    pass
                else:
                    print("Activation selected not available")
                    sys.exit()

        if batch_norm_value_out:
            critic_modules = critic_modules + [
                nn.Linear(hidden_units_value[-1], 1),
                nn.BatchNorm1d(1, affine=False),
                # nn.LayerNorm(1, elementwise_affine=False),
            ]
        else:
            critic_modules = critic_modules + [nn.Linear(hidden_units_value[-1], 1)]

        actor_modules = []

        if self.bnflag_input:
            # affine false sould be equal to center and scale to False in TF2
            actor_modules.append(nn.BatchNorm1d(inp_dim, affine=False))
            # actor_modules.append(nn.LayerNorm(inp_dim, elementwise_affine=False))

        # self.input_layer = InputLayer(input_shape=inp_shape)
        # set of hidden layers
        for i in range(len(hidden_units_actor)):
            if i == 0:
                actor_modules.append(nn.Linear(inp_dim, hidden_units_actor[i]))
                if activation == "relu":
                    actor_modules.append(nn.ReLU())
                elif activation == "tanh":
                    actor_modules.append(nn.Tanh())
                elif activation == "linear":
                    pass
                else:
                    print("Activation selected not available")
                    sys.exit()
            else:
                actor_modules.append(
                    nn.Linear(hidden_units_actor[i - 1], hidden_units_actor[i])
                )
                if activation == "relu":
                    actor_modules.append(nn.ReLU())
                elif activation == "tanh":
                    actor_modules.append(nn.Tanh())
                elif activation == "linear":
                    pass
                else:
                    print("Activation selected not available")
                    sys.exit()

        actor_modules = actor_modules + [nn.Linear(hidden_units_actor[-1], out_dim)]

        self.critic = nn.Sequential(*critic_modules)

        self.actor = nn.Sequential(*actor_modules)

        # TODO insert here flag for cont/discrete
        if self.policy_type == "continuous":
            if self.std_transform == "exp":
                self.log_std = nn.Parameter(torch.ones(1, out_dim) * init_std)
            elif self.std_transform == "softplus":
                self.log_std = nn.Parameter(torch.ones(1, out_dim) * init_std)
                self.softplus = nn.Softplus()
                self.init_std = init_std
                self.min_std = min_std

        elif self.policy_type == "discrete":
            pass
        # I could add an initial offset to be sure that output actions are sufficiently low

        # I didn't use apply because I wanted different init for different layer
        # I guess it should be ok anyway but it has to be tested
        self.init_weights()

    def forward(self, x):
        value = self.critic(x)
        if self.policy_type == "continuous":
            mu = self.actor(x)
            if self.std_transform == "exp":
                std = self.log_std.exp().expand_as(
                    mu
                )  # make the tensor of the same size of mu
            elif self.std_transform == "softplus":
                init_const = 1 / self.softplus(
                    torch.tensor(self.init_std - self.min_std)
                )
                std = (
                    self.softplus(self.log_std + init_const) + self.min_std
                ).expand_as(mu)

            dist = Normal(mu, std)

        elif self.policy_type == "discrete":

            logits = self.actor(x)
            # correct when logits contain nans
            # if torch.isnan(logits).sum() > 0:
            #     logits = (
            #         torch.empty(logits.shape)
            #         .uniform_(-0.01, 0.01)
            #         .type(torch.FloatTensor)
            #     )

            dist = Categorical(logits=logits)

        return dist, value

    def init_weights(self):
        # to access module and layer of an architecture
        # https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/2
        for layer in self.actor:  # [:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

        # carefully initialize last layer
        if self.init_last_layers == "rescaled":
            self.actor[-1].weight = torch.nn.Parameter(self.actor[-1].weight * 0.01)
            self.actor[-1].bias = torch.nn.Parameter(self.actor[-1].bias * 0.01)
        elif self.init_last_layers == "normal":
            nn.init.normal_(self.actor[-1].weight, mean=0.0, std=0.01)
            nn.init.constant_(self.actor[-1].bias, 0.01)

        for layer in self.critic:  # [:-2]:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        # carefully initialize last layer
        # for layer in self.critic[-2:]:
        #     if isinstance(layer, nn.Linear):
        #         if self.init_last_layers == 'rescaled':
        #             layer.weight = torch.nn.Parameter(layer.weight * 0.01)
        #             layer.bias = torch.nn.Parameter(layer.bias * 0.01)
        #         elif self.init_last_layers == 'normal':
        #             nn.init.normal_(layer.weight, mean=0.0, std=0.01)
        #             nn.init.constant_(layer.bias, 0.01)


# ############################### DQN ALGORITHM ################################
@gin.configurable()
class PPO:
    def __init__(
        self,
        seed: int,
        gamma: float,
        tau: float,
        clip_param: float,
        vf_c: float,
        ent_c: float,
        input_shape: int,
        hidden_units_value: list,
        hidden_units_actor: list,
        batch_size: int,
        lr: float,
        activation: str,
        optimizer_name: str,
        batch_norm_input: bool,
        batch_norm_value_out: bool,
        action_space,
        policy_type: str,
        init_pol_std: float,
        min_pol_std: float,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps_opt: float = 1e-07,
        lr_schedule: Optional[str] = None,
        exp_decay_rate: Optional[float] = None,
        step_size: Optional[int] = None,
        std_transform: str = "softplus",
        init_last_layers: str = "rescaled",
        rng=None,
        store_diagnostics: bool = False,
        augadv: bool = False,
        eta1: float = 0.1,
        eta2: float = 0.5,
        action_clipping_type: str = 'env',
        tanh_stretching: float = 1.0,
        scale_reward: bool = False,
        modelname: str = "PPO act_crt",
    ):

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device('cpu')
        self.gamma = gamma
        self.tau = tau
        self.clip_param = clip_param
        self.vf_c = vf_c
        self.ent_c = ent_c
        self.batch_size = batch_size
        self.beta_1 = beta_1
        self.eps_opt = eps_opt
        self.action_space = action_space
        self.policy_type = policy_type
        self.augadv= augadv
        self.eta1 = eta1
        self.eta2 = eta2
        self.action_clipping_type = action_clipping_type
        self.tanh_stretching = tanh_stretching
        self.scale_reward = scale_reward


        if gin.query_parameter('%MULTIASSET'):
            self.num_actions = len(gin.query_parameter('%HALFLIFE'))
        else:
            self.num_actions = self.action_space.get_n_actions(policy_type=self.policy_type)
        self.batch_norm_input = batch_norm_input
        self.store_diagnostics = store_diagnostics

        self.experience = {
            "state": [],
            "action": [],
            "reward": [],
            "log_prob": [],
            "value": [],
            "returns": [],
            "advantage": [],
            "mw_action": [],
            "rl_action": [],
        }

        self.model = PPOActorCritic(
            seed,
            input_shape,
            activation,
            hidden_units_value,
            hidden_units_actor,
            self.num_actions,
            batch_norm_input,
            batch_norm_value_out,
            self.policy_type,
            init_pol_std,
            min_pol_std,
            std_transform,
            init_last_layers,
            modelname,
        )

        self.model.to(self.device)

        self.optimizer_name = optimizer_name
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=(beta_1, beta_2),
                eps=eps_opt,
                weight_decay=0,
                amsgrad=False,
            )
        elif optimizer_name == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=lr,
                alpha=beta_1,
                eps=eps_opt,
                weight_decay=0,
                momentum=0,
                centered=False,
            )

        if lr_schedule == "step":
            self.scheduler = StepLR(
                optimizer=self.optimizer, step_size=step_size, gamma=exp_decay_rate
            )

        elif lr_schedule == "exponential":
            self.scheduler = ExponentialLR(
                optimizer=self.optimizer, gamma=exp_decay_rate
            )
        else:
            self.scheduler = None

        if self.policy_type == 'continuous':
            self.std_hist = []
            self.entropy_hist = []
        elif self.policy_type == 'discrete':
            self.logits_hist = []
            self.entropy_hist = []
        self.total_loss = []
        self.policy_objective = []
        self.value_objective = []
        self.entropy_objective = []
        self.total_loss_byepoch = []
        self.policy_objective_byepoch = []
        self.value_objective_byepoch = []
        self.entropy_objective_byepoch = []
        self.std_hist_byepoch = []


    def train(self, state, action, old_log_probs, return_, advantage, mw_action, rl_action, iteration, epoch, episode):
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

        self.model.train()
        dist, value = self.model(state)
        entropy = dist.entropy().mean()
        if self.policy_type == 'continuous':
            new_log_probs = dist.log_prob(action)
            # self.std_hist.append(self.model.log_std.exp().detach().cpu().numpy().ravel())
            # self.entropy_hist.append(entropy.detach().cpu().numpy().ravel())
        elif self.policy_type == 'discrete':
            new_log_probs = dist.log_prob(action.reshape(-1)).reshape(-1,1)
            # self.logits_hist.append(dist.logits.detach().cpu().numpy())
            # self.entropy_hist.append(entropy.detach().cpu().numpy().ravel())

        if self.augadv:
            # reg = np.log(new_log_probs - old_log_probs 
            reg1 = np.log(mw_action/rl_action).nan_to_num(0.0)

            mask = torch.sign(mw_action)*torch.sign(rl_action)
            mask[mask==1.0] = 0.0
            mask[mask==-1.0] = 1.0
            reg2 = mask * torch.log(torch.abs(mw_action) + torch.abs(rl_action))

            advantage = advantage - self.eta1 * reg1 - self.eta2 * reg2


        
        ratio = (new_log_probs - old_log_probs).exp()  # log properties
        surr1 = ratio * advantage
        surr2 = (
            torch.clamp(ratio, 1.0 / (1 + self.clip_param), 1.0 + self.clip_param)
            * advantage
        )

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = (return_ - value).pow(2).mean()

        # the loss is negated in order to be maximized
        self.loss = self.vf_c * critic_loss + actor_loss - self.ent_c * entropy

        if self.store_diagnostics:
            self.total_loss.append(self.loss.detach().cpu())
            self.policy_objective.append(actor_loss.detach().cpu())
            self.value_objective.append(self.vf_c * critic_loss.detach().cpu())
            self.entropy_objective.append(-self.ent_c * entropy.detach().cpu())
            self.std_hist.append(self.model.log_std.exp().detach().cpu().numpy().ravel())
            if iteration == (self.n_batches-1):
                self.total_loss_byepoch.append(np.mean(self.total_loss))
                self.policy_objective_byepoch.append(np.mean(self.policy_objective))
                self.value_objective_byepoch.append(np.mean(self.value_objective))
                self.entropy_objective_byepoch.append(np.mean(self.entropy_objective))
                self.std_hist_byepoch.append(self.std_hist)
                self.epoch_counter += 1
                if epoch == (self.n_epochs - 1):
                    self.writer.add_scalar("Train/total_loss", np.mean(self.total_loss_byepoch), 
                    episode)
                    self.writer.add_scalar("Train/policy_objective", np.mean(self.policy_objective_byepoch), 
                    episode)
                    self.writer.add_scalar("Train/value_objective", np.mean(self.value_objective_byepoch), 
                    episode)
                    self.writer.add_scalar("Train/entropy_objective", np.mean(self.entropy_objective_byepoch), 
                    episode)
                    self.writer.add_scalar("Train/policy_std", np.mean(self.std_hist_byepoch), 
                                        episode)
                    self.writer.flush()

                    self.total_loss_byepoch = []
                    self.policy_objective_byepoch = []
                    self.value_objective_byepoch = []
                    self.entropy_objective_byepoch = []
            
                self.total_loss = []
                self.policy_objective = []
                self.value_objective = []
                self.entropy_objective = []


        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

    def act(self, states):
        # useful when the states are single dimensional
        self.model.eval()
        # make 1D tensor to 2D
        with torch.no_grad():
            states = torch.from_numpy(states).float().unsqueeze(0)
            states = states.to(self.device)
            return self.model(states)

    def compute_gae(self, next_value, recompute_value=False):

        if recompute_value:
            self.model.eval()
            with torch.no_grad():
                _, values = self.model(
                    torch.Tensor(np.array(self.experience["state"])).to(self.device)
                )
            self.experience["value"] = [
                np.array(v, dtype=float) for v in values.detach().cpu().tolist()
            ]
            # for i in range(len(self.experience["value"])):
            #     _, value = self.act(self.experience["state"][i])
            #     self.experience["value"][i] = value.detach().cpu().numpy().ravel()
        
        rewards = self.experience["reward"]
        values = self.experience["value"]




        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] - values[step]
            gae = delta + self.gamma * self.tau * gae
            returns.insert(0, gae + values[step])

        # add estimated returns and advantages to the experience
        self.experience["returns"] = returns

        advantage = [returns[i] - values[i] for i in range(len(returns))]
        self.experience["advantage"] = advantage

    # add way to reset experience after one rollout
    def add_experience(self, exp):
        for key, value in exp.items():
            self.experience[key].append(value)

    def reset_experience(self):

        self.experience = {
            "state": [],
            "action": [],
            "reward": [],
            "log_prob": [],
            "value": [],
            "returns": [],
            "advantage": [],
            "mw_action": [],
            "rl_action": [],
        }

    def ppo_iter(self):
        # pick a batch from the rollout
        states = np.asarray(self.experience["state"])
        actions = np.asarray(self.experience["action"])
        log_probs = np.asarray(self.experience["log_prob"])
        returns = np.asarray(self.experience["returns"])
        advantage = np.asarray(self.experience["advantage"])
        mw_actions = np.asarray(self.experience["mw_action"])
        rl_actions = np.asarray(self.experience["rl_action"])

        len_rollout = states.shape[0]
        ids = self.rng.permutation(len_rollout)
        ids = np.array_split(ids, len_rollout // self.batch_size)
        self.n_batches = len(ids)
        for i in range(len(ids)):

            yield (
                torch.from_numpy(states[ids[i], :]).float().to(self.device),
                torch.from_numpy(actions[ids[i], :]).float().to(self.device),
                torch.from_numpy(log_probs[ids[i], :]).float().to(self.device),
                torch.from_numpy(returns[ids[i], :]).float().to(self.device),
                torch.from_numpy(advantage[ids[i], :]).float().to(self.device),
                torch.from_numpy(mw_actions[ids[i], :]).float().to(self.device),
                torch.from_numpy(rl_actions[ids[i], :]).float().to(self.device),
            )

    def add_tb_diagnostics(self,path,n_epochs):
        log_dir = os.path.join(path, "tb")
        self.writer = SummaryWriter(log_dir)
        self.epoch_counter = 0
        self.n_epochs = n_epochs

    def getBack(self, var_grad_fn):
        print(var_grad_fn)
        for n in var_grad_fn.next_functions:
            if n[0]:
                try:
                    tensor = getattr(n[0], "variable")
                    print(n[0])
                    print("Tensor with grad found:", tensor)
                    print(" - gradient:", tensor.grad)
                    print()
                except AttributeError as e:
                    self.getBack(n[0])

    def save_diagnostics(self,path):
        if self.policy_type == 'continuous':
            np.save(os.path.join(path, "std_hist"), np.array(self.std_hist))
            np.save(os.path.join(path, "entropy_hist"), np.array(self.entropy_hist))
        elif self.policy_type == 'discrete':
            np.save(os.path.join(path, "logits_hist"), np.array(self.logits_hist, dtype=object))
            np.save(os.path.join(path, "entropy_hist"), np.array(self.entropy_hist))

if __name__ == "__main__":

    model = PPOActorCritic(
        seed=12,
        input_shape=(8,),
        hidden_units=[256, 128],
        num_actions=1,
        batch_norm_input=True,
        std=0.0,
        modelname="PPO",
    )
