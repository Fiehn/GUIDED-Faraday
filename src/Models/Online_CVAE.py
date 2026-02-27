from src.Models.CVAE import *

# ==============
# Online CVAE class for streaming data - inherits from CVAE but adds online updating capabilities
# ==============
class OnlineCVAE(CVAE):
    def __init__(
        self,
        input_dim: int,
        *,
        conditioner: Conditioner,
        latent_dim: int = 32,
        distribution_dict: dict = None,
        prior_type: str = None,
        normalizer: Normalizer = None,
        hierarchy_config: HierarchicalConfig = None,
        **kwargs,
    ):
        super().__init__(input_dim=input_dim, conditioner=conditioner, latent_dim=latent_dim, distribution_dict=distribution_dict, prior_type=prior_type, **kwargs)
        
        # For smart meter data: L=24 hours, F=1 feature
        # input_dim = L * F = 24
        self.sequence_length = 24
        self.num_features = max(1, input_dim // self.sequence_length)
        
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = Normalizer('online_log1p')
        
        # Convert conditioners to online versions
        self._convert_to_online_conditioners()
    
    def _convert_to_online_conditioners(self):
        """Convert regular UserProfiler conditioners to online versions."""
        from Data.Conditioner import OnlineUserProfiler, UserProfiler
        
        for tag, item in self.conditioner.condition_items.items():
            if isinstance(item, UserProfiler) and not isinstance(item, OnlineUserProfiler):
                # Convert to online version
                online_profiler = OnlineUserProfiler(
                    n_topics=item.n_topics,
                    num_clusters=item.num_clusters,
                    decay=0.95,  # Set reasonable default
                    update_frequency=3,  # Update every 10 batches
                    random_state=item.random_state,
                    path=item.path,
                    verbose=item.verbose
                )
                
                # Transfer any existing state
                if hasattr(item, 'cluster_centers_') and item.cluster_centers_ is not None:
                    online_profiler.cluster_centers_ = item.cluster_centers_
                    online_profiler.kmeans_initialized = True
                if hasattr(item, 'gamma_cache'):
                    online_profiler.gamma_cache = item.gamma_cache.copy()
                if hasattr(item, 'user_profiles_cache'):
                    online_profiler.user_profiles_cache = item.user_profiles_cache.copy()
                if hasattr(item, 'is_fitted'):
                    online_profiler.is_fitted = item.is_fitted
                
                # Replace in conditioner
                self.conditioner.condition_items[tag] = online_profiler
    
    def train_core(self, batch, optim, beta=1.0, weighted_loss=0.0, gradient_clip=None, gradient_clip_kwargs=None, **_):
        self.train()

        # 1. Unpack batch
        if len(batch) == 4:
            x, meter_id, timestamps, month_key = batch
        elif len(batch) == 3:
            x, meter_id, timestamps = batch
        elif len(batch) == 2:
             x, meter_id = batch
        else:
            x = batch
        
        # 2. Handle input shapes: convert to 3D for RevIN if needed
        if x.dim() == 2:
            # Assume input is [B, L*F] and reshape to [B, L, F] 
            B = x.shape[0]
            x_3d = x.reshape(B, self.sequence_length, self.num_features)
        elif x.dim() == 3:
            x_3d = x
            B, L, F = x_3d.shape
            # Update dimensions if they don't match initialization
            if F != self.num_features:
                self.num_features = F
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}")

        # 4. Flatten normalized input for the model
        if x_3d.dim() == 3:
            B, L, F = x_3d.shape  
            x_flat = x_3d.reshape(B, L * F)
        else:
            x_flat = x_3d
            
        # 5. Compute conditions using ORIGINAL (unnormalized) input
        # This is important because user profilers expect original energy patterns
        if x.dim() == 2:
            x_for_conditions = x.reshape(B, self.sequence_length, self.num_features)
        else:
            x_for_conditions = x
            
        c = self.conditioner.get_conditions(x_for_conditions, meter_id=meter_id, timestamps=timestamps)
        c = self.move_to_device(c, device=x.device)

        # 6. Forward pass on normalized data
        recon_params, q_params = self.forward(x_flat, c, num_samples=1)

        # 7. Compute loss on normalized data
        optim.zero_grad(set_to_none=True)
        loss_dict = self.loss(x_flat, recon_params, q_params, beta=beta, weighted_loss=weighted_loss)
        loss_dict["loss"].backward()

        # 8. Gradient clipping
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_kwargs.get("max_norm", 1.0))

        # 9. Optimizer step
        optim.step()
        return loss_dict
    
    @torch.no_grad()
    def validate(self, valloader, num_samples=1, beta=1.0, device="cpu", month_key=None):
        self.eval()
        agg = {"loss": 0.0, "recon_loglikelihood": 0.0, "kl_divergence": 0.0}
        num_batches = 0

        for x, meter_id, timestamps, _ in valloader:
            x = x.to(device)
            
            if x.dim() == 2:
                B = x.shape[0]
                x_3d = x.reshape(B, self.sequence_length, self.num_features)
            else:
                x_3d = x

            if x_3d.dim() == 3:
                B, L, F = x_3d.shape
                x_flat = x_3d.reshape(B, L * F)
            else:
                x_flat = x_3d

            c = self.conditioner.get_conditions(x_3d, meter_id=meter_id, timestamps=timestamps)
            c = self.move_to_device(c, device=device)

            recon_params, q_params = self.forward(x_flat, c, num_samples=1)
            loss_dict = self.loss(x_flat, recon_params, q_params, beta=beta)

            for k in agg:
                agg[k] += float(loss_dict[k])
            num_batches += 1

        for k in agg:
            agg[k] /= max(num_batches, 1)
        return agg
