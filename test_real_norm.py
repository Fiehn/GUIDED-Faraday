import torch
import sys
import os
from src.models.cvae import CVAE
from src.visualization.plot_styling import PlotStyle
from src.visualization.visualizer import Visualizer
import traceback
from torch.utils.data import DataLoader
from src.data.normalizer import Normalizer

try:
    print("Loading model and normalizer...")
    model, normalizer = CVAE.load_from_checkpoint('models/CVAE_DictionaryCoder_20260318_120522_vamp_softmax_userprofiler/CVAE_DictionaryCoder_20260318_120522_final.pt')
    if normalizer is None:
        print("Normalizer could not be loaded, creating a fallback one")
        normalizer = Normalizer(method='log1p')
        normalizer.fitted = True
        normalizer.method = 'log1p'
        normalizer.mean = torch.zeros(1)
        normalizer.std = torch.ones(1)

    os.makedirs('test_log', exist_ok=True)
    plot_style = PlotStyle()
    vis = Visualizer(plot_style=plot_style, log_dir='test_log', model=model, normalizer=normalizer, conditioner=model.conditioner, device='cuda:0')
    
    class DummyDS(torch.utils.data.Dataset):
        def __len__(self): return 5
        def __getitem__(self, idx): return torch.randn(24, 1), torch.tensor(idx), torch.tensor(1600000000 + idx*86400)
    
    test_loader = DataLoader(DummyDS(), batch_size=5)
    print("Testing pca_compared...")
    vis.pca_compared(test_loader)
    print("Testing visualize_synthetic...")
    vis.visualize_synthetic(num_samples=5)
    
except Exception as e:
    traceback.print_exc()
