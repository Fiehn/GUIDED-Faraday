import torch
import sys
import os
from src.model.cvae import CVAE
from src.visualization.plot_styling import PlotStyle
from src.visualization.visualizer import Visualizer
import traceback

class MockNorm:
    def inverse_transform(self, x): return x
    def transform(self, x): return x

try:
    model, _ = CVAE.load_from_checkpoint('models/CVAE_DictionaryCoder_20260318_120522_vamp_softmax_userprofiler/CVAE_DictionaryCoder_20260318_120522_final.pt')
    os.makedirs('test_log', exist_ok=True)
    plot_style = PlotStyle()
    vis = Visualizer(plot_style=plot_style, log_dir='test_log', model=model, normalizer=MockNorm(), conditioner=model.conditioner, device='cuda:0')
    
    x = torch.randn(2, 24, 1).to('cuda:0')
    meter_id = torch.tensor([1, 2])
    timestamps = torch.tensor([1600000000, 1600000000]).to('cuda:0')
    
    print('Testing get_conditions')
    c = vis.model.conditioner.get_conditions(x, meter_id, timestamps).to(vis.device)
    vis.generator.generate(num_samples=x.shape[-1], conditions=c)
    print('Done pca')

    print('Testing random conditions')
    vis.visualize_synthetic(num_samples=5)
    print('Done synthetic')
except Exception as e:
    traceback.print_exc()

