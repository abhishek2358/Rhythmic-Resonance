# Rhythmic Resonance: Music Generation using LSTM with Local Attention

## Abstract

This project introduces a novel method employing deep learning to produce single-line melodies from provided beats, enabling individuals of all skill levels to craft their unique musical pieces. The utilization of LSTM with Local Attention for this purpose offers a wide range of melodies with rich harmony and well-defined structure. Through this project, individuals can easily engage in music composition simply by tapping on their keyboards.

## Getting Started

To get started, clone this repository and install the required packages:
```sh
git clone https://github.com/abhishek2358/Rhythmic-Resonance-Music-Generation-using-LSTM-with-Local-Attention.git
```

## Training
```sh
python train.py -m lstm_attn
```

## Generating Melodies from Beats

Execute the predict_stream.py script to generate a predicted sequence of notes and store it as a MIDI file.

Specify the path to the checkpoint file for the model using the -c followed by a string.

The resulting MIDI file will be named according to the -o.

```sh
python predict_stream.py -c ./.project_data/snapshots/my_checkpoint.pth
```
