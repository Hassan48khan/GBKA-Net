"""Central configuration for GBKA-Net experiments.

Every value here matches the paper. Command-line flags in train.py / evaluate.py
override these defaults.
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # data
    img_size: int = 256
    input_channels: int = 1          # grayscale ultrasound
    num_classes: int = 1

    # model
    embed_dims: tuple = (256, 320, 512)
    depths: tuple = (1, 1, 1)
    factor: int = 16                 # GKMSA group count
    deep_supervision: bool = True

    # ablation toggles (Tables 5-6)
    no_gkmsa_gate: bool = False      # disable global-to-local gate in GKMSA
    no_gbaf_gate: bool = False       # disable evidence gate in GBAF
    no_deepsup: bool = False         # supervise only the final head
    no_kan: bool = False             # replace KAN layers with MLPs

    # optimization
    optimizer: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    batch_size: int = 32
    max_epochs: int = 300
    early_stop_patience: int = 20

    # loss
    seg_gamma: float = 5.0           # boundary-weighting strength
    w_boundary: float = 1.0          # lambda_b

    # misc
    seed: int = 42
    num_workers: int = 4
