from .q_learning import QLearningAgent
from .sarsa import SARSAAgent
from .dqn import DQNAgent
from .reinforce import ReinforceAgent

AGENT_REGISTRY = {
    "q_learning": QLearningAgent,
    "sarsa": SARSAAgent,
    "dqn": DQNAgent,
    "reinforce": ReinforceAgent,
}

__all__ = ["AGENT_REGISTRY", "QLearningAgent", "SARSAAgent", "DQNAgent", "ReinforceAgent"]
