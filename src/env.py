#!/usr/bin/env python

import random
import numpy as np
from gym import spaces
from osim.env import L2M2019Env

from catalyst.rl.core import EnvironmentSpec
from catalyst.rl.utils import extend_space

from .env_wrappers import EnvNormalizer


BIG_NUM = np.iinfo(np.int32).max


class SkeletonEnvWrapper(EnvironmentSpec):
    def __init__(
        self,
        history_len=1,
        frame_skip=1,
        reward_scale=1,
        reload_period=None,
        action_mean=None,
        action_std=None,
        visualize=False,
        mode="train",
        **params
    ):
        super().__init__(visualize=visualize, mode=mode)

        env = L2M2019Env(**params, visualize=visualize)
        env = EnvNormalizer(env)
        self.env = env

        self._history_len = history_len
        self._frame_skip = frame_skip
        self._visualize = visualize
        self._reward_scale = reward_scale
        self._reload_period = reload_period or BIG_NUM
        self.episode = 0

        self.action_mean = np.array(action_mean) \
            if action_mean is not None else None
        self.action_std = np.array(action_std) \
            if action_std is not None else None

        self._prepare_spaces()

    @property
    def history_len(self):
        return self._history_len

    @property
    def observation_space(self) -> spaces.space.Space:
        return self._observation_space

    @property
    def state_space(self) -> spaces.space.Space:
        return self._state_space

    @property
    def action_space(self) -> spaces.space.Space:
        return self._action_space

    def _prepare_spaces(self):
        self._observation_space = self.env.observation_space
        self._action_space = self.env.action_space

        self._state_space = extend_space(
            self._observation_space, self._history_len
        )

    def _process_action(self, action):
        if self.action_mean is not None \
                and self.action_std is not None:
            action = action * (self.action_std + 1e-8) + self.action_mean
        return action

    def reset(self):
        if self.episode % self._reload_period == 0:
            difficulty = random.randint(0, 2)
            seed = random.randrange(BIG_NUM)
            self.env.change_model(difficulty=difficulty, seed=seed)

        self.episode += 1
        observation = self.env.reset()
        return observation

    def step(self, action):
        reward = 0
        action = self._process_action(action)
        for i in range(self._frame_skip):
            observation, r, done, info = self.env.step(action)
            if self._visualize:
                self.env.render()
            reward += r
            if done:
                break
        info["raw_reward"] = reward
        reward *= self._reward_scale
        return observation, reward, done, info
