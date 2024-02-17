from abc import ABC, abstractmethod
from typing import List, Tuple
import torch
from utils.model import localAttnLSTM

from utils.constants import NOTE_START


class DistributionGenerator(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def initial_state(self, hint: List[int]) -> dict:
        pass

    @abstractmethod
    def proceed(self, state: dict, prev_note: int) -> Tuple[dict, torch.Tensor]:
        pass



class LocalAttnLSTMDistribution(DistributionGenerator):
    def __init__(self, model: localAttnLSTM, x, device):
        self.model = model
        self.device = device
        self.x = x
        self.context, self.encoder_state = self.model.encoder(x)

    def initial_state(self, hint: List[int]) -> dict:
        super().initial_state(hint)
        hint_shifted = [NOTE_START] + hint[:-1]
        state = {
            "position": 0,
            "memory": (self.encoder_state[0].reshape(1, 1, -1), self.encoder_state[1].reshape(1, 1, -1)),
        }
        for i in range(len(hint)):
            state, _ = self.proceed(state, hint_shifted[i])
        return state

    def proceed(self, state: dict, prev_note: int) -> Tuple[dict, torch.Tensor]:
        super().proceed(state, prev_note)
        position = state["position"]
        memory = state["memory"]
        context_curr = self.context[position].reshape(1, 1, -1)

        y_prev = torch.tensor(prev_note).reshape(1, 1).to(self.device)
        scores, memory = self.model.decoder.forward(y_prev, context_curr, memory)

        scores = scores.squeeze(0)
        scores = torch.nn.functional.softmax(scores, dim=1)
        scores = scores.squeeze(0)
        return {"position": position + 1, "memory": memory}, scores