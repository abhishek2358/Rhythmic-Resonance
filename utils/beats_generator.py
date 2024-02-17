from pynput import keyboard
import numpy as np
import time
from pathlib import Path


from preprocess.prepare import convert_start_end_to_beats
from enum import Enum
from sys import platform



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


class Event(Enum):
    PRESS = 1
    RELEASE = 0

def create_beat():
    TAP_KEYS = {keyboard.KeyCode.from_char('z'), keyboard.KeyCode.from_char('x'), keyboard.Key.space}
    ENTER_KEY = keyboard.Key.enter

    events = []
    pressing_key = None
    base_time = time.time()
    def on_press(key):

        nonlocal events, pressing_key
        if key in TAP_KEYS and key != pressing_key:
            curr_time = time.time() - base_time
            if pressing_key is not None:
                events.append((Event.RELEASE, curr_time))
            events.append((Event.PRESS, curr_time))
            pressing_key = key

    def on_release(key):
        nonlocal events, pressing_key
        if key == ENTER_KEY:
            # Stop listener
            curr_time = time.time() - base_time
            if pressing_key is not None:
                events.append((Event.RELEASE, curr_time))
            return False
        elif key in TAP_KEYS and key == pressing_key:
            events.append((Event.RELEASE, time.time() - base_time))
            pressing_key = None

    suppress = True if platform == "win32" else False
    with keyboard.Listener(on_press=on_press, on_release=on_release, suppress=suppress) as listener:
        listener.join()

    start_time = []
    end_time = []
    num_pressed = 0
    for event, timestamp in events:
        if event == Event.PRESS:
            start_time.append(timestamp)
            if num_pressed > 0:
                end_time.append(timestamp)
            num_pressed += 1
        else:
            if num_pressed == 1:
                end_time.append(timestamp)
            num_pressed -= 1
    beat_sequence = convert_start_end_to_beats(np.array(start_time), np.array(end_time))

    paths = createStructure()
    file_name = "last_recorded.npy"
    file_path = paths.beats_rhythms_dir / file_name
    np.save(file_path, beat_sequence)

    return beat_sequence