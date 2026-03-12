from src.models.cvae import *

# ==============
# SET-CVAE class for training on multiple datasets sequentially
# ==============
 
# Notes:
# - the encoder and decoder are really the only changes tbh, and some data handling
# - The prior they use is MoG, we should just add a new prior type for that and reuse the same code
# - the data handling should really just be additions to the base models


class SetCVAE(CVAE):
    ... # TODO: do it



# ==============
# NN Blocks
# ==============

# === MAB (Multihead Attention Block) ===
class MAB(nn.Module):
    """
    Multihead Attention Block as used in Set Transformers.
    """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super(MAB, self).__init__()
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_V = dim_V
        self.num_heads = num_heads

        

# === ISAB (Induced Set Attention Block) ===
class ISAB(nn.Module):
    ...

# === ABL (Attentive Bottleneck Layer) ===
class ABL(nn.Module):
    ...

