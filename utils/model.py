from typing import Optional
import torch
import toml
from preprocess.dataset import DataProcess
import torch.utils.data
from pathlib import Path
from torch import nn
from torch.nn import functional as F
CONFIG_PATH = "./config.toml"



class EncodeLocalAttn(nn.Module):
    def __init__(self, hidden_dim):
        super(EncodeLocalAttn, self).__init__()
        self.duration_fc = nn.Linear(2, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional = True)

    def forward(self, x):
        x = self.duration_fc(x)
        x, encoder_state = self.encoder(x)
        return x, (encoder_state[0][:1], encoder_state[1][:1])

class DecodeLocalAttn(nn.Module):
    def __init__(self, hidden_dim, num_notes, dropout_p = 0.1):
        super(DecodeLocalAttn, self).__init__()
        self.note_embed = nn.Embedding(num_notes, hidden_dim)
        self.combine_fc = nn.Linear(hidden_dim * 3, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.notes_output = nn.Linear(hidden_dim, num_notes)

    def forward(self, tgt, context, memory = None):
        tgt = self.note_embed(tgt)
        tgt = torch.cat((tgt, context), dim=2)
        tgt = self.combine_fc(tgt)
        tgt = F.relu(tgt)
        tgt = self.dropout(tgt)
        tgt, memory = self.rnn(tgt, memory)
        tgt = self.notes_output(tgt)
        return tgt, memory
    
class localAttnLSTM(nn.Module):
    def __init__(self, num_notes, hidden_dim, dropout_p = 0.1):
        super(localAttnLSTM, self).__init__()
        self.encoder = EncodeLocalAttn(hidden_dim)
        self.decoder = DecodeLocalAttn(hidden_dim, num_notes, dropout_p)
        self.num_notes = num_notes
    
    def forward(self, x, tgt):
        context, encoder_state = self.encoder(x)
        output, _ = self.decoder(tgt, context, encoder_state)
        return output

    def loss_function(self, pred, target):

        criterion = nn.CrossEntropyLoss()
        target = target.flatten() 
        pred = pred.reshape(-1, pred.shape[-1])
        loss = criterion(pred, target)
        return loss

    def clip_gradients_(self, max_value):
        torch.nn.utils.clip_grad.clip_grad_value_(self.parameters(), max_value)


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

def train(model_name: str, n_epochs: int, device: str, n_files:int=-1, snapshots_freq:int=10, checkpoint: Optional[str] = None):

    model = localAttnLSTM(num_notes=128, hidden_dim=512,dropout_p=0.5).to(device)
    print(model)

    dataset = DataProcess(128, 123)
    
    dataset.load("mastero")
    dataset = dataset.subset_remove_short()
    if n_files > 0:
        dataset = dataset.subset(n_files)

    training_data, val_data = dataset.train_val_split(666, 0.1)
    print(f"Training data: {len(training_data)}")
    print(f"Validation data: {len(val_data)}")

    train_loader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)
    epochs_start = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    paths = createStructure()
    

    for epoch in range(epochs_start, n_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_seq = batch["beats"].to(device)
            target_seq = batch["notes"].long().to(device)
            target_prev_seq = batch["notes_shifted"].long().to(device)
            output = model(input_seq, target_prev_seq)
            train_loss = model.loss_function(output, target_seq)
            train_loss.backward()
            model.clip_gradients_(5.0)
            optimizer.step()
        
        model.eval()
        for batch in val_loader:
            input_seq = batch["beats"].to(device)
            target_seq = batch["notes"].long().to(device)
            target_prev_seq = batch["notes_shifted"].long().to(device)
            with torch.no_grad():
                output = model(input_seq, target_prev_seq)
                val_loss = model.loss_function(output, target_seq)
        
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Train Loss: {:.4f}, Val Loss: {:.4f}".format(train_loss.item(), val_loss.item()))

        
    model_file = "lstmlocalattn.pth"
    model_path = paths.snapshots_dir / model_file
    torch.save({'model': model.state_dict(),'n_epochs': n_epochs,}, model_path)
    print(f'Checkpoint Saved at {model_path}')
    return model
