from dataclasses import dataclass


@dataclass
class TrainingArgs:
    lr: float = 1e-4
    lr_schedule: str = "linear_with_warmup"
    lr_warmup_steps: int = 1000
    gradient_clip_val: float = 1.0
    checkpoint_path: str = "checkpoints"
    from_checkpoint: str = None
    output_path: str = "outputs"
    run_name: str = None
    wandb_mode: str = "online"
    wandb_project: str = "mpm-conversational"
    wandb_dir: str = "wandb"
    n_workers: int = None
    n_steps: int = 20000
    batch_size: int = 8
    seed: int = 0
    dataset: str = "data/fischer"
    log_every_n_steps: int = 100
    do_save: bool = True
    save_onnx: bool = False
    eval_only: bool = False
    eval_every_n_steps: int = 1000
    save_every_n_steps: int = 1000
    push_to_hub: bool = False
    hub_repo: str = None
    dryrun: bool = False
    drop_channel_prob: float = 0.05


@dataclass
class CollatorArgs:
    overwrite: bool = False
    bin_size: int = 128
    mask_proportion: float = 0.475
    mask_proportion_tolerance: float = 0.05
    mask_length: int = 10
    mask_length_max: int = None
    drop_input_prob: float = 0.05
    max_length: int = 2048
    name: str = "fischer"
    pitch_min: float = 50
    pitch_max: float = 600
    energy_min: float = 0
    energy_max: float = 1
    vad_min: float = 0
    vad_max: float = 1


@dataclass
class ModelArgs:
    n_layers: int = 8
    n_heads: int = 2
    kernel_size: int = 7
    filter_size: int = 256
    hidden_dim: int = 512
    dropout: float = 0.1
