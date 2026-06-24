from __future__ import annotations
from abc import ABC, abstractmethod
import torch

Tensor = torch.Tensor

class BaseTrigger(ABC):
    """
    Abstract class for backdoor triggers.
    """
    def __init__(self, position, size, pattern, alpha=1.0):
        """
        Initializes the trigger's core attributes.
        Args:
            # UPDATED: Clarified the coordinate system convention.
            position (Tuple[int, int]): Top-left corner (x, y) of the trigger,
                                       where x is columns from left and y is rows from top.
            size (Tuple[int, int]): Size (width, height) of the trigger.
            pattern: The trigger pattern itself (e.g., a color value, a small tensor).
            alpha (float): The blending factor for the trigger's opacity.
        """
        self.position = position
        self.size = size
        self.pattern = pattern
        self.alpha = alpha

    @abstractmethod
    def apply(self, image: Tensor) -> Tensor:
        """Applies the trigger to a given image."""
        pass