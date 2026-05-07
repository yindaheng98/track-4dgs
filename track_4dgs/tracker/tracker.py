from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Optional

import torch

from .types import Query


class AbstractPointTracker(metaclass=ABCMeta):
    """Base class for point trackers over one camera sequence."""

    def to(self, device) -> 'AbstractPointTracker':
        return self

    @abstractmethod
    def __call__(
            self,
            query: Query,
            ground_truth_images: Sequence[torch.Tensor],
            ground_truth_image_masks: Sequence[Optional[torch.Tensor]]) -> torch.Tensor:
        """Return tracked pixel coordinates with shape ``[D, N, 2]``."""
        raise NotImplementedError
