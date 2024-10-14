import os.path as osp
import pprint

from colorama import Fore, Style
from tabulate import tabulate
from yacs.config import CfgNode as CN


def create_cfg():
    cfg = CN()
    cfg._BASE_ = None
    cfg.PROJECT_NAME = "Makeup Transfer with Diffusion"
    cfg.PROJECT_DIR = None

    # ##### Model setup #####
    cfg.MODEL = CN()
    cfg.MODEL.IN_CHANNELS = 3
    cfg.MODEL.OUT_CHANNELS = cfg.MODEL.IN_CHANNELS
    cfg.MODEL.LAYERS_PER_BLOCK = 2
    cfg.MODEL.BASE_DIM = 192
    cfg.MODEL.LAYER_SCALE = [1, 1, 2, 2, 4, 4]
    cfg.MODEL.PRETRAINED = None  # "pretrained/pretrained.pkl"
    cfg.MODEL.LABEL_DIM = 0

    # ###### Training set ######
    cfg.TRAIN = CN()
    cfg.TRAIN.MAKEUP = None

    # Log and save
    cfg.TRAIN.RESUME = None
    cfg.TRAIN.IMAGE_SIZE = 256
    cfg.TRAIN.LOG_INTERVAL = 20
    cfg.TRAIN.SAVE_INTERVAL = 10000
    cfg.TRAIN.SAMPLE_INTERVAL = 10000
    cfg.TRAIN.ROOT = None

    # Training iteration
    cfg.TRAIN.BATCH_SIZE = 4
    cfg.TRAIN.NUM_WORKERS = 4
    cfg.TRAIN.MAX_ITER = 350000
    cfg.TRAIN.GRADIENT_ACCUMULATION_STEPS = 16
    cfg.TRAIN.MIXED_PRECISION = "no"
    cfg.TRAIN.GRAD_NORM = 1.0

    # EMA setup
    cfg.TRAIN.EMA_MAX_DECAY = 0.9999
    cfg.TRAIN.EMA_INV_GAMMA = 1.0
    cfg.TRAIN.EMA_POWER = 0.75

    # Optimizer
    cfg.TRAIN.LR = 0.0001
    cfg.TRAIN.LR_WARMUP = 1000

    # Diffusion setup
    cfg.TRAIN.TIME_STEPS = 1000
    cfg.TRAIN.SAMPLE_STEPS = cfg.TRAIN.TIME_STEPS
    cfg.TRAIN.NOISE_SCHEDULER = CN()
    # ///// for linear start \\\\\\\
    cfg.TRAIN.NOISE_SCHEDULER.BETA_START = 1e-4
    cfg.TRAIN.NOISE_SCHEDULER.BETA_END = 0.02
    # ///// for linear end \\\\\\\
    cfg.TRAIN.NOISE_SCHEDULER.TYPE = "linear"
    cfg.TRAIN.NOISE_SCHEDULER.PRED_TYPE = "epsilon"

    # ======= Evaluation set =======
    cfg.EVAL = CN()
    cfg.EVAL.BATCH_SIZE = 16
    cfg.EVAL.SAMPLE_STEPS = 1000
    cfg.EVAL.ETA = 0.01
    cfg.EVAL.REFINE_STEPS = 0
    cfg.EVAL.REFINE_ITERATIONS = 0
    cfg.EVAL.SCHEDULER = "ddpm"

    return cfg


def merge_possible_with_base(cfg: CN, config_path):
    with open(config_path, "r") as f:
        new_cfg = cfg.load_cfg(f)
    if "_BASE_" in new_cfg:
        cfg.merge_from_file(osp.join(osp.dirname(config_path), new_cfg._BASE_))
    cfg.merge_from_other_cfg(new_cfg)


def split_into(v):
    res = "(\n"
    for item in v:
        res += f"    {item},\n"
    res += ")"
    return res


def pretty_print_cfg(cfg):
    def _indent(s_, num_spaces):
        s = s_.split("\n")
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(num_spaces * " ") + line for line in s]
        s = "\n".join(s)
        s = first + "\n" + s
        return s

    r = ""
    s = []
    for k, v in sorted(cfg.items()):
        seperator = "\n" if isinstance(v, CN) else " "
        attr_str = "{}:{}{}".format(
            str(k),
            seperator,
            pretty_print_cfg(v) if isinstance(v, CN) else pprint.pformat(v),
        )
        attr_str = _indent(attr_str, 2)
        s.append(attr_str)
    r += "\n".join(s)
    return r


def show_config(cfg):
    table = tabulate(
        {"Configuration": [pretty_print_cfg(cfg)]}, headers="keys", tablefmt="fancy_grid"
    )
    print(f"{Fore.BLUE}", end="")
    print(table)
    print(f"{Style.RESET_ALL}", end="")
