
 
# ==============
# Cyclical Beta Annealing Scheduler
# Reference: "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing"
# (Fu et al., 2019) https://arxiv.org/abs/1903.10145
# ==============
class CyclicalBetaScheduler:
    """Cyclical annealing schedule for the KL weight (beta) in VAE training.

    Each cycle consists of a linear warmup from 0 to *beta_max* followed by a
    constant phase at *beta_max*.  The proportion of each cycle spent warming
    up is controlled by *ratio* (default 0.5 = half warmup, half constant).

    Parameters
    ----------
    total_steps : int
        Total number of training steps (epochs * batches_per_epoch).
    n_cycles : int
        Number of full annealing cycles over training (default 4).
    ratio : float
        Fraction of each cycle devoted to the linear warmup (0-1, default 0.5).
    beta_max : float
        Maximum beta value reached at the top of each cycle (default 1.0).
    """

    def __init__(self, total_steps: int, n_cycles: int = 4, ratio: float = 0.5, beta_max: float = 1.0):
        assert total_steps > 0, "total_steps must be > 0"
        assert n_cycles >= 1, "n_cycles must be >= 1"
        assert 0.0 < ratio <= 1.0, "ratio must be in (0, 1]"
        self.total_steps = total_steps
        self.n_cycles = n_cycles
        self.ratio = ratio
        self.beta_max = beta_max
        self._step = 0

    def step(self) -> float:
        """Advance one step and return the current beta value."""
        beta = self._get_beta(self._step)
        self._step += 1
        return beta

    def _get_beta(self, step: int) -> float:
        """Compute beta for a given global step."""
        # How far through training are we (0..1)
        tau = step / self.total_steps
        # Which cycle are we in, and where inside that cycle (0..1)
        cycle_progress = (tau * self.n_cycles) % 1.0
        if cycle_progress < self.ratio:
            # Linear warmup phase
            return self.beta_max * (cycle_progress / self.ratio)
        else:
            # Constant phase
            return self.beta_max

    def state_dict(self):
        return {"step": self._step, "total_steps": self.total_steps,
                "n_cycles": self.n_cycles, "ratio": self.ratio, "beta_max": self.beta_max}

    @classmethod
    def from_state_dict(cls, d):
        obj = cls(d["total_steps"], d["n_cycles"], d["ratio"], d["beta_max"])
        obj._step = d["step"]
        return obj

    def __repr__(self):
        return (f"CyclicalBetaScheduler(total_steps={self.total_steps}, n_cycles={self.n_cycles}, "
                f"ratio={self.ratio}, beta_max={self.beta_max}, current_step={self._step})")


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.mode = mode

    def __call__(self, train_metric, val_metric, epoch=None):
        score = -val_metric if self.mode == 'min' else val_metric

        if train_metric is not None and val_metric is not None:
            if abs(train_metric - val_metric) < self.min_delta * 10:
                # Overfitting detected - Stop training
                self.early_stop = True
                return False

        if self.best_score is None:
            # First time call
            self.best_score = score
            if epoch is not None:
                self.best_epoch = epoch
            return True
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False                
        else:
            self.best_score = score
            self.counter = 0
            if epoch is not None:
                self.best_epoch = epoch
            return True
    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    
# === Multivariate Student T distribtion, not in PyTorch by default, so we implement it here ===
import math
import torch
import torch.nn as nn
from torch.distributions import Distribution, constraints
from torch.distributions.utils import _standard_normal, lazy_property

class MultivariateStudentT(Distribution):
    # Constraints: df > 0, scale_tril is lower triangular with positive diagonal
    arg_constraints = {'df': constraints.positive, 'loc': constraints.real_vector, 'scale_tril': constraints.lower_cholesky}
    support = constraints.real_vector
    has_rsample = True

    def __init__(self,df,loc,covariance_matrix=None,scale_tril=None,validate_args=None):
        # Handle covariance input (similar to MultivariateNormal)
        if scale_tril is None:
            if covariance_matrix is not None:
                # Cholesky decomposition to get scale_tril
                scale_tril = torch.linalg.cholesky(covariance_matrix)
            else:
                raise ValueError("Either covariance_matrix or scale_tril must be specified")
        self.df = df
        self.loc = loc
        self.scale_tril = scale_tril
        self._event_shape = self.loc.size()[-1:]
        self._batch_shape = self.loc.size()[:-1]

        super().__init__(batch_shape=self._batch_shape, event_shape=self._event_shape, validate_args=validate_args)

    @lazy_property
    def precision_matrix(self):
        # inverse of covariance matrix (cholesky)
        identity = torch.eye(self.scale_tril.size(-1), device=self.scale_tril.device)
        return torch.cholesky_solve(identity, self.scale_tril)
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        
        # Dimensions
        d = self.loc.size(-1)
        df = self.df

        # Mahalanobis distance squared: (x-mu)^T Sigma^-1 (x-mu)
        diff = value - self.loc
        M = torch.linalg.solve_triangular(self.scale_tril, diff.unsqueeze(-1), upper=False).squeeze(-1)  # Solve L y = diff
        mahalanobis = M.pow(2).sum(-1)

        # Log probability for multivariate Student's t-distribution
        log_gamma_term = torch.lgamma((df + d) * 0.5) - torch.lgamma(df * 0.5)
        log_det_term = -0.5 * d * math.log(math.pi * df) - torch.log(self.scale_tril.diagonal(dim1=-2, dim2=-1)).sum(-1)
        log_poly_term = -0.5 * (df + d) * torch.log1p(mahalanobis / df)

        return log_gamma_term + log_det_term + log_poly_term
    
    def rsample(self, sample_shape = torch.Size()):
        # Reparameterization trick for sampling from multivariate Student's t-distribution
        # x = mu + (scale * z) / sqrt(u / df) where z ~ N(0, I) and u ~ Chi2(df)
        shape = self._extended_shape(sample_shape)

        # Sample z from standard normal
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)

        # Sample u from chi-squared distribution (Chi2(df) is equivalent to Gamma(df/2, 0.5))
        u = torch.distributions.Gamma(self.df / 2, 0.5).rsample(shape[:-1])

        # Construct affine transformation
        z_transformed = (self.scale_tril @ eps.unsqueeze(-1)).squeeze(-1)

        # Scale by sqrt((df - 2) / df) for correct variance and divide by sqrt(u/df) for the t-distribution scaling
        scale_factor = torch.sqrt(u / self.df).unsqueeze(-1)

        return self.loc + z_transformed / scale_factor



