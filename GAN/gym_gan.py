#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
import argparse
import gym
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision. utils as vutils
from tensorboardX import SummaryWriter


class InputWrapper(gym.ObservationWrapper):

    def __init__(self, *args):
        super().__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.img_size = 64
        self.observation_space = gym.spaces.Box(self.observation(old_space.low), self.observation(old_space.high), dtype=np.float32)

    def observation(self, observation):
        # resize img from 210x160 (standard atari) to 64x64
        new_obs = cv2.resize(observation, (self.img_size, self.img_size))
        # change (height, width, channels) to pytorch CONV-convention (channels, height, width)
        new_obs = np.moveaxis(new_obs, 2, 0)
        # change img from byte to float in range [0, 1]
        return new_obs.astype(np.float32)


class Discriminator(nn.Module):

    def __init__(self, input_shape):
        super().__init__()
        
        # probability that Discriminator thinks the input img is from a real dataset
        self.conv_pipe = nn.Sequential(
                nn.Conv2d(in_channels=input_shape[0], out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0),
                nn.Sigmoid()
                )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        return conv_out.view(-1, 1).squeeze(dim=1)


class Generator(nn.Module):

    def __init__(self, output_shape):
        super().__init__()
        
        # deconvolve input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
                nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=64, out_channels=output_shape[0], kernel_size=4, stride=2, padding=1),
                nn.Tanh()
                )
    
    def forward(self, x):
        return self.pipe(x)


def iterate_batches(envs, batch_size=16):
    batch = [env.reset() for env in envs]
    env_gen = iter(lambda: random.choice(envs), None)
    
    while True:
        e = next(env_gen)

        state, reward, done, _ = e.step(e.action_space.sample())
        
        if np.mean(state) > 0.01:
            batch.append(state)

        if len(batch) == batch_size:
            # normalize input to [-1, 1]
            batch_np = np.array(batch, dtype=np.float32) * 2.0 / 255.0 - 1.0
            yield torch.tensor(batch_np)
            batch.clear()

        if done:
            e.reset()


if __name__ == "__main__":
    log = gym.logger
    log.set_level(gym.logger.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable cuda.')
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda else 'cpu')

    envs = [InputWrapper(gym.make(name)) for name in ('Breakout-v0', 'AirRaid-v0', 'Pong-v0')]
    
    input_shape = envs[0].observation_space.shape

    discriminator = Discriminator(input_shape=input_shape).to(device)
    generator = Generator(output_shape=input_shape).to(device)

    loss = nn.BCELoss()

    gen_optimizer = optim.Adam(params=generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(params=discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    writer = SummaryWriter()

    gen_losses = []
    dis_losses = []
    iter_no = 0

    batch_size = 16

    true_labels_v = torch.ones(batch_size, dtype=torch.float32, device=device)
    fake_labels_v = torch.zeros(batch_size, dtype=torch.float32, device=device)

    for batch_v in iterate_batches(envs):
        # generate more fake samples of dimension (batch, filters, x, y)
        gen_input_v = torch.FloatTensor(batch_size, 100, 1, 1).normal_(0, 1).to(device)
        batch_v = batch_v.to(device)
        gen_output_v = generator(gen_input_v)

        # train discriminator
        dis_optimizer.zero_grad()
        dis_output_true_v = discriminator(batch_v)
        dis_output_fake_v = discriminator(gen_output_v.detach())
        dis_loss = loss(dis_output_true_v, true_labels_v) + loss(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        # train generator
        gen_optimizer.zero_grad()
        dis_output_v = discriminator(gen_output_v)
        gen_loss_v = loss(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        iter_no += 1
        print(iter_no)
        if iter_no % 100 == 0:
            log.info('Iter %d: gen_loss=%.3e, dis_loss=%.3e', iter_no, np.mean(gen_losses), np.mean(dis_losses))
            writer.add_scalar('gen_loss', np.mean(gen_losses), iter_no)
            writer.add_scalar('dis_loss', np.mean(dis_losses), iter_no)
            gen_losses = []
            dis_losses = []
        if iter_no % 1000 == 0:
            writer.add_image('fake', vutils.make_grid(gen_output_v.data[:64], normalize=True), iter_no)
            writer.add_image('real', vutils.make_grid(batch_v.data[:64], normalize=True), iter_no)
