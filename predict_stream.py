import argparse

import numpy as np
import preprocess.dataset
import torch
import toml
from utils.constants import NOTE_MAP

from utils.model import CONFIG_PATH, localAttnLSTM
from utils.render import render_midi
from utils.sample import beam_search, stochastic_search
from pathlib import Path


class createStructure:
    def __init__(self):
        self.cache_dir = Path(".project_data")
        self.cache_dir.mkdir(exist_ok=True)
        self.downloads_dir = self.cache_dir / "downloads"
        self.downloads_dir.mkdir(exist_ok=True)
        self.snapshots_dir = self.cache_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)
        self.prepared_data_dir = self.cache_dir / "prepared_data"
        self.prepared_data_dir.mkdir(exist_ok=True)
        self.midi_outputs_dir = self.cache_dir / "midi_outputs"
        self.midi_outputs_dir.mkdir(exist_ok=True)
        self.tensorboard_dir = self.cache_dir / "tensorboard"
        self.tensorboard_dir.mkdir(exist_ok=True)
        self.beats_rhythms_dir = self.cache_dir / "beats_rhythms"
        self.beats_rhythms_dir.mkdir(exist_ok=True)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Save Predicted Notes Sequence to Midi')
    parser.add_argument('-m','--model_name', type=str)
    parser.add_argument('-c','--checkpoint_path', type=str)
    parser.add_argument('-o','--midi_filename', type=str, default="output.mid")
    parser.add_argument('-d','--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-s','--source', type=str, default="interactive")
    parser.add_argument('-t','--profile', type=str, default="default")

    main_args = parser.parse_args()
    model_name = main_args.model_name
    checkpoint_path = main_args.checkpoint_path
    midi_filename = main_args.midi_filename
    device = main_args.device
    source = main_args.source
    profile = main_args.profile

    config = toml.load(CONFIG_PATH)
    global_config = config['global']
    model_config = config["model"][main_args.model_name]

    paths = createStructure()
    if main_args.source == 'interactive':
        from utils.beats_generator import create_beat
        X = create_beat()
        X[0][0] = 2.
        X = np.array(X, dtype=np.float32) 
    elif main_args.source == 'dataset':
        dataset = preprocess.dataset.BeatsRhythmsDataset(64)
        dataset.load(global_config['dataset'])
        idx = np.random.randint(0, len(dataset))
        X = dataset.beats_list[idx][:64]
    else:
        with open(main_args.source, 'rb') as f:
            X = np.load(f, allow_pickle=True)
            X[0][0] = 2.
    X = np.array(X, dtype=np.float32)
    model = localAttnLSTM(num_notes=128, hidden_dim=512,dropout_p=0.5).to(device)
    model.eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    n_epochs = checkpoint['n_epochs']
    print(model)
    profile = config["sampling"][profile]
    try:
        hint = [NOTE_MAP[h] for h in profile["hint"]]
    except KeyError:
        print(f"some note in {profile['hint']} not found in NOTE_MAP")
        exit(1)
    if profile["strategy"] == "stochastic":
        notes = stochastic_search(model, X, hint, device, profile["top_p"], profile["top_k"], profile["repeat_decay"], profile["temperature"])
    elif profile["strategy"] == "beam":
        notes = beam_search(model, X, hint, device, profile["repeat_decay"], profile["num_beams"], profile["beam_prob"], profile["temperature"])
    else:
        raise NotImplementedError(f"strategy {profile['strategy']} not implemented")
    print(notes)
    midi_paths = paths.midi_outputs_dir / main_args.midi_filename
    render_midi(X, notes, midi_paths)