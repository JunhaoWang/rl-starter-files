import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
import gym
# note grid_seq_2_idx_seq should only be used when FullyObsWrapper is used
from utils.misc import grid_seq_2_idx_seq

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False, arch = 0):
        super().__init__()



        self.arch1 = (arch ==1)
        self.arch2 = (arch ==2)
        self.arch3 = (arch ==3)
        # Collect set of trajectory sequence information (state_sequence, state_space_cardinality, grid_shape,
        # state_value_value) encoded by discrete state. Note it should only be used for discrete state when FullyObsWrapper
        # is used.
        self.traj_info_set = []
        self.obs_list = []

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 32, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        #self.embedding_size = 256
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        if self.arch1:
            self.linear1   = nn.Sequential(
                                nn.Linear(self.embedding_size, self.embedding_size//2),
                                nn.Tanh())
            self.embedding_size = self.embedding_size//2

        elif self.arch2:
            self.linear1   = nn.Sequential(
                                nn.Linear(self.embedding_size, self.embedding_size//2),
                                nn.Tanh(),
                                nn.Linear(self.embedding_size//2, self.embedding_size//4),
                                nn.Tanh())
            self.embedding_size = self.embedding_size//4
        elif self.arch3:
            self.linear1   = nn.Sequential(
                                nn.Linear(self.embedding_size, self.embedding_size//2),
                                nn.Tanh(),
                                nn.Linear(self.embedding_size//2, self.embedding_size//4),
                                nn.Tanh(),
                                nn.Linear(self.embedding_size//4, 64),
                                nn.Tanh())
            self.embedding_size = 64

        # Define actor's model
        if isinstance(action_space, gym.spaces.Discrete):
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, 3)
            )
        else:
            raise ValueError("Unknown action space: " + str(action_space))

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        # note grid_seq_2_idx_seq should only be used when FullyObsWrapper is used
        # idx_seq_info contains [state_idx_seq, state_idx_max, grid_shape, state_value_seq]
        idx_seq_info = list(grid_seq_2_idx_seq(obs.image))
        self.obs_list.append(obs.image)

        self.obs_list.append(obs.image)

        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.arch1 or self.arch2 or self.arch3:
            x = self.linear1(x)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        idx_seq_info.append(value.tolist())

        self.traj_info_set.append(tuple(idx_seq_info))

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
