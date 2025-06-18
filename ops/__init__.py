# %%
from pathlib import Path
import torch
import logging


build_dir = Path(__file__).parent.parent / "build"
for lib in build_dir.iterdir():
    if lib.name.startswith("lib"):
        for file in lib.glob("*.so"):
            logging.info(f"Loading custom op: {file}")
            torch.ops.load_library(file)

