"""Quick verification of all changes."""
import torch
import torch.nn as nn
import tempfile
import shutil
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.checkpoint_manager import CheckpointManager, EMAModel
from src.data.cached_dataset import CachedSRDataset

passed = 0
failed = 0

def check(name, cond):
    global passed, failed
    if cond:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name}")
        failed += 1

# ================================================
print("\n=== Test 1: EMA state_dict round-trip ===")
model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
ema = EMAModel(model, decay=0.9995)
x = torch.randn(4, 10)
y = model(x); y.sum().backward()
for p in model.parameters():
    p.data -= 0.01 * p.grad; p.grad = None
ema.update(model)
sd = ema.state_dict()
check("state_dict keys", 'shadow' in sd and 'decay' in sd)
check("decay preserved", sd['decay'] == 0.9995)
ema2 = EMAModel(model, decay=0.5)
ema2.load_state_dict(sd)
check("decay restored", ema2.decay == 0.9995)
check("shadow values match", all(torch.allclose(ema.shadow[k], ema2.shadow[k]) for k in ema.shadow))

# ================================================
print("\n=== Test 2: CheckpointManager EMA save/load ===")
tmp = Path(tempfile.mkdtemp())
mgr = CheckpointManager(str(tmp / 'ckpt'), keep_best_k=2, save_every=1)
optimizer = torch.optim.Adam(model.parameters())
mgr.save_checkpoint(epoch=5, model=model, optimizer=optimizer, metrics={'psnr': 33.5}, is_best=True, ema=ema)
ckpt_files = list((tmp / 'ckpt').glob('*.pth'))
check("checkpoint saved", len(ckpt_files) > 0)
ckpt = torch.load(ckpt_files[0], weights_only=False)
check("ema_state_dict in checkpoint", 'ema_state_dict' in ckpt)

# ================================================
print("\n=== Test 3: Best checkpoint amnesia fix ===")
mgr.save_checkpoint(epoch=10, model=model, optimizer=optimizer, metrics={'psnr': 34.0}, is_best=True, ema=ema)
mgr2 = CheckpointManager(str(tmp / 'ckpt'), keep_best_k=2, save_every=1)
check("best_checkpoints restored", len(mgr2.best_checkpoints) > 0)
check("best metric correct", mgr2.best_checkpoints[0][0] == 34.0)
check("is_best(33.0) = False", not mgr2.is_best(33.0))
check("is_best(35.0) = True", mgr2.is_best(35.0))
shutil.rmtree(tmp)

# ================================================
print("\n=== Test 4: CachedSRDataset old format ===")
tmp1 = Path(tempfile.mkdtemp())
for i in range(3):
    stem = f'img_{i:03d}'
    torch.save({'outputs': {'hat': torch.randn(1,3,256,256)}, 'features': {'hat': torch.randn(1,180,64,64)}, 'lr': torch.randn(3,64,64), 'hr': torch.randn(3,256,256), 'filename': stem}, tmp1 / f'{stem}_hat_part.pt')
    torch.save({'outputs': {'dat': torch.randn(1,3,256,256), 'nafnet': torch.randn(1,3,256,256)}, 'features': {'dat': torch.randn(1,180,64,64), 'nafnet': torch.randn(1,64,64,64)}, 'filename': stem}, tmp1 / f'{stem}_rest_part.pt')
ds1 = CachedSRDataset(str(tmp1), augment=False, repeat_factor=1)
s = ds1[0]
check("old: 3 experts", set(s['expert_imgs'].keys()) == {'hat','dat','nafnet'})
check("old: lr shape", s['lr'].shape == (3,64,64))
shutil.rmtree(tmp1)

# ================================================
print("\n=== Test 5: CachedSRDataset new 5-crop format ===")
tmp2 = Path(tempfile.mkdtemp())
for i in range(2):
    for p in range(5):
        stem = f'img_{i:03d}_p{p}'
        torch.save({'outputs': {'drct': torch.randn(1,3,256,256)}, 'features': {'drct': torch.randn(1,180,64,64)}, 'lr': torch.randn(3,64,64), 'hr': torch.randn(3,256,256), 'filename': stem}, tmp2 / f'{stem}_drct_part.pt')
        torch.save({'outputs': {'grl': torch.randn(1,3,256,256), 'nafnet': torch.randn(1,3,256,256)}, 'features': {'grl': torch.randn(1,180,64,64), 'nafnet': torch.randn(1,64,64,64)}, 'filename': stem}, tmp2 / f'{stem}_rest_part.pt')
ds2 = CachedSRDataset(str(tmp2), augment=False, repeat_factor=1)
s2 = ds2[0]
check("new: drct->hat mapping", 'hat' in s2['expert_imgs'])
check("new: grl->dat mapping", 'dat' in s2['expert_imgs'])
check("new: 10 samples (2x5)", len(ds2) == 10)
check("new: features mapped", set(s2['expert_feats'].keys()) == {'hat','dat','nafnet'})
shutil.rmtree(tmp2)

# ================================================
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed")
print(f"{'='*50}")
if failed == 0:
    print("ALL TESTS PASSED!")
sys.exit(0 if failed == 0 else 1)
