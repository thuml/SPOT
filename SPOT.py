import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import os
from vae import VAE
from utils import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init_(m, init_w=3e-3):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-init_w, init_w)
        m.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, dropout=None, hidden_dim=256, init_w=None):
        super(Actor, self).__init__()

        if dropout:
            self.l1 = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Dropout(dropout))
            self.l2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout))
        else:
            self.l1 = nn.Linear(state_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

        if init_w:
            weights_init_(self.l3, init_w=init_w)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.l3(a)
        if self.max_action is not None:
            action = self.max_action * torch.tanh(a)
        else:
            action = a
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, init_w=None):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

        if init_w:
            weights_init_(self.l3, init_w=init_w)
            weights_init_(self.l6, init_w=init_w)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class SPOT_TD3(object):
    def __init__(
            self,
            vae: VAE,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            beta=0.5,
            lambd=1.0,
            lr=3e-4,
            actor_lr=None,
            without_Q_norm=False,
            # density estimation
            num_samples=1,
            iwae=False,
            # actor-critic
            actor_hidden_dim=256,
            critic_hidden_dim=256,
            actor_dropout=0.1,
            actor_init_w=None,
            critic_init_w=None,
            # finetune
            lambd_cool=False,
            lambd_end=0.2,
    ):
        self.total_it = 0

        # Actor
        self.actor = Actor(state_dim, action_dim, max_action, dropout=actor_dropout,
                           hidden_dim=actor_hidden_dim, init_w=actor_init_w).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr or lr)

        # Critic
        self.critic = Critic(state_dim, action_dim, hidden_dim=critic_hidden_dim, init_w=critic_init_w).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # TD3
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        # density estimation
        self.vae = vae
        self.beta = beta
        self.num_samples = num_samples
        self.iwae = iwae
        self.without_Q_norm = without_Q_norm

        # support constraint
        self.lambd = lambd

        # fine-tuning
        self.lambd_cool = lambd_cool
        self.lambd_end = lambd_end

    def select_action(self, state):
        with torch.no_grad():
            self.actor.eval()
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action = self.actor(state).cpu().data.numpy().flatten()
            self.actor.train()
            return action

    def train(self, replay_buffer: ReplayBuffer, batch_size=256, logger=None):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Log
        logger.log('train/critic_loss', critic_loss, self.total_it)

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)

            if self.iwae:
                neg_log_beta = self.vae.iwae_loss(state, pi, self.beta, self.num_samples)
            else:
                neg_log_beta = self.vae.elbo_loss(state, pi, self.beta, self.num_samples)

            if self.without_Q_norm:
                actor_loss = - Q.mean() + self.lambd * neg_log_beta.mean()
            else:
                actor_loss = - Q.mean() / Q.abs().mean().detach() + self.lambd * neg_log_beta.mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Log
            logger.log('train/Q', Q.mean(), self.total_it)
            logger.log('train/actor_loss', actor_loss, self.total_it)
            logger.log('train/neg_log_beta', neg_log_beta.mean(), self.total_it)
            logger.log('train/neg_log_beta_max', neg_log_beta.max(), self.total_it)

            #  kill for diverging
            if Q.mean().item() > 1e4:
                exit(0)

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_online(self, replay_buffer: ReplayBuffer, batch_size=256, logger=None):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Log
        logger.log('train/critic_loss', critic_loss, self.total_it)

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)

            if self.iwae:
                neg_log_beta = self.vae.iwae_loss(state, pi, self.beta, self.num_samples)
            else:
                neg_log_beta = self.vae.elbo_loss(state, pi, self.beta, self.num_samples)

            if self.lambd_cool:
                lambd = self.lambd * max(self.lambd_end, (1.0 - self.total_it / 1000000))
                logger.log('train/lambd', lambd, self.total_it)
            else:
                lambd = self.lambd

            if self.without_Q_norm:
                actor_loss = - Q.mean() + lambd * neg_log_beta.mean()
            else:
                actor_loss = - Q.mean() / Q.abs().mean().detach() + lambd * neg_log_beta.mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Log
            logger.log('train/Q', Q.mean(), self.total_it)
            logger.log('train/actor_loss', actor_loss, self.total_it)
            logger.log('train/neg_log_beta', neg_log_beta.mean(), self.total_it)
            logger.log('train/neg_log_beta_max', neg_log_beta.max(), self.total_it)

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, model_dir):
        torch.save(self.critic.state_dict(), os.path.join(model_dir, f"critic_s{str(self.total_it)}.pth"))
        torch.save(self.critic_target.state_dict(), os.path.join(model_dir, f"critic_target_s{str(self.total_it)}.pth"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(
            model_dir, f"critic_optimizer_s{str(self.total_it)}.pth"))

        torch.save(self.actor.state_dict(), os.path.join(model_dir, f"actor_s{str(self.total_it)}.pth"))
        torch.save(self.actor_target.state_dict(), os.path.join(model_dir, f"actor_target_s{str(self.total_it)}.pth"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(
            model_dir, f"actor_optimizer_s{str(self.total_it)}.pth"))

    def load(self, model_dir, step=1000000):
        self.critic.load_state_dict(torch.load(os.path.join(model_dir, f"critic_s{str(step)}.pth")))
        self.critic_target.load_state_dict(torch.load(os.path.join(model_dir, f"critic_target_s{str(step)}.pth")))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(model_dir, f"critic_optimizer_s{str(step)}.pth")))

        self.actor.load_state_dict(torch.load(os.path.join(model_dir, f"actor_s{str(step)}.pth")))
        self.actor_target.load_state_dict(torch.load(os.path.join(model_dir, f"actor_target_s{str(step)}.pth")))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(model_dir, f"actor_optimizer_s{str(step)}.pth")))
