

from .dinov2 import DINOv2
from .gmm_seg import GMMSegHead

from .misc import (
    LinearHead,
    MLP,
    MultiScaleMLP,
    CascadedMLP,
    create_model
)



from easydict import EasyDict as edict

MODELS = edict(
    DINOv2=DINOv2,
    GMMSegHead=GMMSegHead,
    LinearHead=LinearHead
)

BACKBONE_OUTPUT_DIMS = edict(
    dinov2_vits14_reg=384,
    dinov2_vitb14_reg=768,
    dinov2_vitl14_reg=1024,
    dinov2_vitg14_reg=1536
)