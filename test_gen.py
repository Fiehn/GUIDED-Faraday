import torch
import sys
import os
from src.model.cvae import CVAE
from src.visualization.plot_styling import PlotStyle
from src.visualization.visualizer import Visualizer
import traceback
from torch.utils.data import DataLoader

class MockNorm:
    def inverse_transform(self, x): return x
    def transform(self, x): return x

try:
    model, _ = CVAE.load_from_checkpoint('models/CVAE_DictionaryCoder_20260318_120522_vamp_softmax_userprofiler/CVAE_DictionaryCoder_20260318_120522_final.pt')
    os.makedirs('test_log', exist_ok=True)
    plot_style = PlotStyle()
    vis = Visualizer(plot_style=plot_style, log_dir='test_log', model=model, normalizer=MockNorm(), conditioner=model.conditioner, device='cuda:0')
    
    class DummyDS(torch.utils.data.Dataset):
        def __len__(self): return 2
        def __getitem__(self, idx): return torch.randn(24, 1), torch.tensor(idx), torch.tensor(1600000000 + idx*86400)
    
    test_loader = DataLoader(DummyDS(), batch_size=2)
    print("Testing visualizer")
    vis.pca_compared(test_loader)
    
except Exception as e:
    traceback.print_exc()
