#Author : Paarth Sharma
#File Name : benchmark_vjepa2.py
#Project Name : swiftJEPA
#Creation Date : 16th September 2026
#Modification Date : 16th September 2026
#Description : Baseline latency harness for the official V-JEPA 2 encoder. 

# In-built libraries
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# External Libraries
import numpy as np
import torch
import src.hub.backbones as backbones

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VJEPA2_REPO = ROOT / "vjepa2"
# this lets src.hub.backbones resolve to the vendored vjepa2 clone.
sys.path.insert(0, str(VJEPA2_REPO))

# override with the public checkpoint host before any hub load.
PUBLIC_CKPT_URL = "https://dl.fbaipublicfiles.com/vjepa2"

# name to (builder attr in src.hub.backbones, img_size)
MODELS = {
    "vitl": ("vjepa2_vit_large", 256),
    "vith": ("vjepa2_vit_huge", 256),
    "vitg": ("vjepa2_vit_giant", 256),
    "vitg-384": ("vjepa2_vit_giant_384", 384),
}

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args():
    p = argparse.ArgumentParser(description="V-JEPA 2 inference benchmark")
    p.add_argument("--model", default="vitl", choices=MODELS.keys())
    p.add_argument("--frames", type=int, default=16, help="video frames (multiple of tubelet size 2)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", default="fp16", choices=DTYPES.keys())
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--pretrained", action="store_true", help="download real weights")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    return p.parse_args()


# build the encoder straight from the vendored vjepa2 hub code.
def build_encoder(args):
    
    # the vjepa2 repo points this url at localhost for testing at HEAD. we override it here so the real checkpoint host gets used instead.
    backbones.VJEPA_BASE_URL = PUBLIC_CKPT_URL

    builder_name, img_size = MODELS[args.model]
    
    builder = getattr(backbones, builder_name)
    encoder, _predictor = builder(pretrained=args.pretrained, num_frames=args.frames)

    # the builder returns an encoder and a predictor since JEPA pretraining needs both. inference only needs the encoder so we drop the predictor here.
    del _predictor
    
    encoder.eval().to(args.device)
    
    return encoder, img_size


# Register CUDA-event hooks on patch_embed, block and norm.
class LayerTimer:
    
    def __init__(self, encoder, device):
        
        self.device = device
        self.stages = [("patch_embed", encoder.patch_embed)]
        self.stages.extend([(f"block_{i:02d}", block) for i, block in enumerate(encoder.blocks)])
        self.stages.append(("norm", encoder.norm))
        self.records = []  # list per iteration
        self._current = None
        self._handles = []
        
        # pre hook records the start event and post hook records the end event, together each stage gets its own timed pair.
        for name, mod in self.stages:
            self._handles.append(mod.register_forward_pre_hook(self.pre(name)))
            self._handles.append(mod.register_forward_hook(self.post(name)))

    def pre(self, name):

        # pytorch calls hooks with only module and input so we bake the stage name in here.
        def hook(_mod, _inp):

            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            
            # this is a list and not a tuple because post fills in the second slot later.
            self._current[name] = [evt, None]

        return hook

    def post(self, name):
        def hook(_mod, _inp, _out):
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            self._current[name][1] = evt
        return hook

    def start_iter(self):
        self._current = {}

    def end_iter(self):
        
        self.records.append(self._current)
        
        self._current = None

    def elapsed_ms(self):
        
        out = []
        
        for rec in self.records:
            out.append({name: s.elapsed_time(e) for name, (s, e) in rec.items()})
            
        return out

    def remove(self):
        
        for handle in self._handles:
            handle.remove()

def main():
    pass


if __name__ == "__main__":
    main()
