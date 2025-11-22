"""Weighted Gradient Explainer - extends Integrated Gradients with baseline weighting."""

from .ig_explainer import IGExplainer


class WGExplainer(IGExplainer):
    """Weighted Gradient Explainer.
    
    This class extends the Integrated Gradients explainer to support weighted
    baseline attribution. It uses the same interface as IGExplainer but is
    designed to work with multiple baselines that can be weighted based on
    their contribution to the explanation.
    
    The weighting is typically done externally based on deletion metrics.
    """
    
    def __init__(self, model, data, batch_size=50, local_smoothing=0):
        """Initialize the Weighted Gradient explainer.
        
        Args:
            model: PyTorch model or tuple of (model, layer) for interim layer attribution
            data: Background data tensor or list of tensors (typically multiple baselines)
            batch_size: Batch size for gradient computation
            local_smoothing: Standard deviation for local smoothing noise
        """
        super().__init__(model, data, batch_size, local_smoothing)

