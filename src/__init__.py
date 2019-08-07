from catalyst.rl import registry

from src.env import SkeletonEnvWrapper
from src.actor import SkeletonActor
from src.critic import SkeletonStateCritic, SkeletonStateActionCritic

registry.Environment(SkeletonEnvWrapper)
registry.Agent(SkeletonActor)
registry.Agent(SkeletonStateCritic)
registry.Agent(SkeletonStateActionCritic)
