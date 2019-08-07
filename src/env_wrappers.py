import numpy as np
from gym import spaces
import gym


class ObservationWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        raise NotImplementedError


class EnvNormalizer(ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
        vector_field_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2, 11, 11),
            dtype=env.observation_space.dtype)
        features_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(97,),
            dtype=env.observation_space.dtype)
        self.observation_space = spaces.Dict({
            "vector_field": vector_field_space,
            "features": features_space
        })

    def _prepare_observation(self, obs_dict):
        # Augmented environment from the L2R challenge
        vector_field = np.array(obs_dict["v_tgt_field"]).reshape(2, 11, 11)

        features = []
        features.append(obs_dict["pelvis"]["height"])
        features.append(obs_dict["pelvis"]["pitch"])
        features.append(obs_dict["pelvis"]["roll"])
        features.append(obs_dict["pelvis"]["vel"][0] / self.env.LENGTH0)
        features.append(obs_dict["pelvis"]["vel"][1] / self.env.LENGTH0)
        features.append(obs_dict["pelvis"]["vel"][2] / self.env.LENGTH0)
        features.append(obs_dict["pelvis"]["vel"][3])
        features.append(obs_dict["pelvis"]["vel"][4])
        features.append(obs_dict["pelvis"]["vel"][5])

        for leg in ["r_leg", "l_leg"]:
            features += obs_dict[leg]["ground_reaction_forces"]
            features.append(obs_dict[leg]["joint"]["hip_abd"])
            features.append(obs_dict[leg]["joint"]["hip"])
            features.append(obs_dict[leg]["joint"]["knee"])
            features.append(obs_dict[leg]["joint"]["ankle"])
            features.append(obs_dict[leg]["d_joint"]["hip_abd"])
            features.append(obs_dict[leg]["d_joint"]["hip"])
            features.append(obs_dict[leg]["d_joint"]["knee"])
            features.append(obs_dict[leg]["d_joint"]["ankle"])
            for MUS in [
                "HAB", "HAD", "HFL",
                "GLU", "HAM", "RF",
                "VAS", "BFSH", "GAS",
                "SOL", "TA"
            ]:
                features.append(obs_dict[leg][MUS]["f"])
                features.append(obs_dict[leg][MUS]["l"])
                features.append(obs_dict[leg][MUS]["v"])
        features = np.array(features)

        return vector_field, features

    def observation(self, obs_dict):
        vector_field, features = self._prepare_observation(obs_dict)

        observation = {
            "vector_field": vector_field,
            "features": features
        }
        return observation
