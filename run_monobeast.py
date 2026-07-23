from contextlib import redirect_stderr, redirect_stdout
import io
import os
import warnings

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", message=".*The version_base parameter is not specified.*")
warnings.filterwarnings("ignore", message=".*Future Hydra versions will no longer change working directory.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")
warnings.filterwarnings("ignore", message=".*Creating a tensor from a list of numpy.ndarrays is extremely slow.*")

# Silence "Loading environment football failed: No module named 'gfootball'" message
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import kaggle_environments

import hydra
import logging
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from torch import multiprocessing as mp
import wandb

from lux_ai.utils import flags_to_namespace
with redirect_stderr(io.StringIO()):
    from lux_ai.torchbeast.monobeast import train


os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=logging.INFO,
)


def get_default_flags(flags: DictConfig) -> DictConfig:
    flags = OmegaConf.to_container(flags)
    # Env params
    flags.setdefault("seed", None)
    flags.setdefault("num_buffers", max(2 * flags["num_actors"], flags["batch_size"] // flags["n_actor_envs"]))
    flags.setdefault("obs_space_kwargs", {})
    flags.setdefault("reward_space_kwargs", {})
    flags.setdefault("env_configuration", {})

    # Training params
    flags.setdefault("use_mixed_precision", True)
    flags.setdefault("discounting", 0.999)
    flags.setdefault("reduction", "mean")
    flags.setdefault("clip_grads", 10.)
    flags.setdefault("checkpoint_freq", 10.)
    flags.setdefault("num_learner_threads", 1)
    flags.setdefault("use_teacher", False)
    flags.setdefault("teacher_baseline_cost", flags.get("teacher_kl_cost", 0.) / 2.)

    # Model params
    flags.setdefault("use_index_select", True)
    if flags.get("use_index_select"):
        logging.info("index_select disables padding_index and is equivalent to using a learnable pad embedding.")

    # Reloading previous run params
    flags.setdefault("load_dir", None)
    flags.setdefault("checkpoint_file", None)
    flags.setdefault("weights_only", False)
    flags.setdefault("n_value_warmup_batches", 0)

    # Miscellaneous params
    flags.setdefault("disable_wandb", False)
    flags.setdefault("debug", False)
    flags.setdefault("log_config", False)
    flags.setdefault("resume_from_local_config", False)

    return OmegaConf.create(flags)


@hydra.main(config_path="conf", config_name="resume_config", version_base=None)
def main(flags: DictConfig):
    cli_conf = OmegaConf.from_cli()
    if flags.get("resume_from_local_config", False) and Path("config.yaml").exists():
        new_flags = OmegaConf.load("config.yaml")
        flags = OmegaConf.merge(new_flags, cli_conf)

    if flags.get("load_dir", None) and not flags.get("weights_only", False):
        # this ignores the local config.yaml and replaces it completely with saved one
        # however, you can override parameters from the cli still
        # this is useful e.g. if you did total_steps=N before and want to increase it
        logging.info("Loading existing configuration, we're continuing a previous run")
        new_flags = OmegaConf.load(Path(flags.load_dir) / "config.yaml")
        # Overwrite some parameters
        new_flags = OmegaConf.merge(new_flags, flags)
        flags = OmegaConf.merge(new_flags, cli_conf)

    flags = get_default_flags(flags)
    if flags.log_config:
        logging.info(OmegaConf.to_yaml(flags, resolve=True))
    else:
        logging.info(
            "Training %s | steps=%s | map=%s | actors=%s envs=%s | teacher=%s",
            flags.name,
            flags.total_steps,
            flags.env_configuration,
            flags.num_actors,
            flags.n_actor_envs,
            flags.use_teacher,
        )
    OmegaConf.save(flags, "config.yaml")
    if not flags.disable_wandb:
        wandb.init(
            config=vars(flags),
            project=flags.project,
            entity=flags.entity,
            group=flags.group,
            name=flags.name,
        )

    flags = flags_to_namespace(OmegaConf.to_container(flags))
    mp.set_sharing_strategy(flags.sharing_strategy)
    train(flags)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
