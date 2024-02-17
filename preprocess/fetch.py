import os
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm
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

def download(filename: str, url: str) -> Optional[Path]:
    paths = createStructure()
    cache_path = paths.downloads_dir / filename
    if not os.path.exists(cache_path):
        with requests.get(url, stream=True) as r:
            if r.status_code == 404:
                return None
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            chunk_size = 1024
            with open(cache_path, "wb") as f:
                print(f"Downloading {filename} from {url}")
                progress = tqdm(total = total_size_in_bytes, unit = 'iB', unit_scale = True, colour="cyan")
                for chunk in r.iter_content(chunk_size=chunk_size):
                    progress.update(len(chunk))
                    f.write(chunk)
                progress.close()
    else:
        print("Using cached: ", filename)
    return cache_path




