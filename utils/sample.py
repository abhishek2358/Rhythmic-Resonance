from typing import List, Tuple
import numpy as np
import torch


from utils.model import localAttnLSTM
from utils.distribution import DistributionGenerator, LocalAttnLSTMDistribution
from tqdm import tqdm

def get_distribution_generator(model, beats, device) -> DistributionGenerator:
    beats = torch.from_numpy(beats).float().to(device)
    return LocalAttnLSTMDistribution(model, beats, device)

def stochastic_step(prev_note: int, distribution: torch.Tensor, top_p: float = 0.9, top_k: int=4, repeat_decay: float = 0.5, temperature = 1.) -> Tuple[int, float]:
    # penalize previous note
    distribution[prev_note] *= (1 - repeat_decay)
    # sample only the top p of the distribution
    sorted_prob, sorted_idx = torch.sort(distribution, descending=True)
    cumsum_prob = torch.cumsum(sorted_prob, dim=0)
    top_p_mask = cumsum_prob < top_p
    top_p_mask[0] = True
    top_p_idx = sorted_idx[top_p_mask][:top_k]
    top_p_distribution = distribution[top_p_idx]
    # normalize the distribution
    top_p_distribution = top_p_distribution / top_p_distribution.sum()
    # apply temperature
    top_p_distribution = top_p_distribution ** (1 / temperature)
    # sample
    sampled_note = int(torch.multinomial(top_p_distribution, 1).item())
    conditional_likelihood = top_p_distribution[sampled_note].item()
    return top_p_idx[sampled_note].item(), conditional_likelihood

def stochastic_search(model, beats: np.ndarray, hint: List[int], device: str, top_p: float= 0.9, top_k:int= 4, repeat_decay: float = 0.5, temperature=1.) -> np.ndarray:
    dist = get_distribution_generator(model, beats, device)
    state = dist.initial_state(hint)
    generated_sequence = hint[:]
    prev_note = generated_sequence[-1]
    progress_bar = tqdm(range(beats.shape[0] - len(hint)), desc="Stochastic search")
    for _ in progress_bar:
        # get the distribution
        state, distribution = dist.proceed(state, prev_note)
        # sample
        sampled_note, _ = stochastic_step(prev_note, distribution, top_p, top_k, repeat_decay, temperature)
        generated_sequence.append(sampled_note)
        prev_note = sampled_note
    return np.array(generated_sequence)

def beam_search(model, beats: np.ndarray, hint: List[int], device: str, repeat_decay: float = 0.5, num_beams: int = 3, beam_prob: float = 0.7, temperature=1.) -> np.ndarray:
    dist = get_distribution_generator(model, beats, device)
    state = dist.initial_state(hint)
    beams = [(hint[:], state, 0)] # (generated_sequence, state, log_likelihood)
    progress_bar = tqdm(range(beats.shape[0] - len(hint)), desc="Beam search")
    for _ in progress_bar:
        beam_choice = np.random.rand()
        if beam_choice < beam_prob:
            new_beams = []
            for beam in beams:
                prev_note = beam[0][-1]
                state, distribution = dist.proceed(beam[1], prev_note)
                # modify the distribution using the repeat_decay
                distribution[prev_note] *= (1 - repeat_decay)
                # sample
                for sampled_note in range(128):
                    new_beam = (beam[0] + [sampled_note], state, beam[2] + np.log(distribution[sampled_note].item()))
                    new_beams.append(new_beam)
            # sort the beams by their likelihood
            new_beams = sorted(new_beams, key=lambda x: x[2], reverse=True)
            # keep only the top num_beams
            beams = new_beams[:num_beams]
        else:
            new_beams = []
            for beam in beams:
                prev_note = beam[0][-1]
                state, distribution = dist.proceed(beam[1], prev_note)
                # sample
                sampled_note, conditional_likelihood = stochastic_step(prev_note, distribution, 1.0, 128, repeat_decay, temperature)
                new_beam = (beam[0] + [sampled_note], state, beam[2] + np.log(conditional_likelihood))
                new_beams.append(new_beam)
            # sort the beams by their likelihood
            new_beams = sorted(new_beams, key=lambda x: x[2], reverse=True)
            # keep only the top num_beams
            beams = new_beams[:num_beams]
    # return the beam with the highest likelihood
    return np.array(beams[0][0])
    


