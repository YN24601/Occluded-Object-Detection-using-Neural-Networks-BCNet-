import wandb
from detectron2.utils import comm


def init_wandb(cfg, resume=False):
    """Initialize a W&B run from cfg.WANDB settings. No-op if disabled or not main process."""
    if not cfg.WANDB.ENABLED:
        return
    if not comm.is_main_process():
        return
    wandb.init(
        entity=cfg.WANDB.ENTITY or None,
        project=cfg.WANDB.PROJECT,
        name=cfg.WANDB.RUN_NAME or None,
        group=cfg.WANDB.GROUP or None,
        notes=cfg.WANDB.NOTES or None,
        tags=cfg.WANDB.TAGS or None,
        resume="allow" if resume else None,
        config={"cfg": cfg},
    )
