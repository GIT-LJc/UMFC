import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# datasets
import datasets.domainnet_cluster    
import datasets.domainnet_preprocessed
import datasets.domainnet_tta
import datasets.imagenet_ars_cluster
import datasets.imagenet_ars_preprocessed

# trainers
import trainers.zsclip
import trainers.umfc_UC
import trainers.umfc_TTC
import trainers.openclip_umfc
import trainers.t3a_clip


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.strong_transforms:
        cfg.TRAINER.FIXMATCH.STRONG_TRANSFORMS = args.strong_transforms

    # if args.gpu:
    #     cfg.USE_CUDA = args.gpu


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "amp"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.DATASET.DOMAIN = 0
    cfg.DATASET.INTERSECTION = False
    cfg.DATASET.Promptconcat = False
    cfg.TRAINER.WEIGHT_DOM = 0.0
    cfg.TRAINER.UNIFORM = False
    cfg.DATASET.NUM_SHOTS_UNLABELED = 96   # shots of unlabeled data by default
    cfg.TRAINER.TOP_K = 0
    cfg.TRAINER.HIGH_THERSHOLD = False
    cfg.TRAINER.GROUND_TRUTH = False
    cfg.TRAINER.CUSTOM_CLIP_PROTOS_RATIO = 0.2
    cfg.DATALOADER.NUM_WORKERS = 8

    cfg.TRAINER.UNLABELED_CLUSTERS = 3    # the number of clusters for unlabeled data
    # cfg.TRAINER.CLIP_THRESHOLD = 0.7   #? the threshold for using clip's predictions
    cfg.TRAINER.CALIBRATE_TEXT = 'norm_ensemble'   # use text calibration for clip's predictions
    cfg.TRAINER.CALIBRATE_IMG_WEIGHT = 0.9   # the weight for the image calibration
    cfg.TRAINER.CALIBRATE_TEXT_WEIGHT = 0.3   # the weight for the text calibration
    cfg.TTC_UPDATE = 'memory'   # the way to update the TTC
    cfg.TRAINER.EMA_DECAY = 0.9   # the decay for EMA update of TTC
    # cfg.DATASET.TEST_DOMAINS = 'all'   # the test domains
    cfg.T3A_FilterK = -1   # the filter value of T3A
    cfg.ARCH = 'ViT-B-16'   # the arch of openclip
    # cfg.TRAINER.TEST_CLUSTERS = 6   # the number of clusters for test data 临时的

    cfg.TRAINER.CLIP_ADAPTER = CN()
    cfg.TRAINER.CLIP_ADAPTER.RATIO = 0.2    # the ratio of resifual connection 
    cfg.TRAINER.WEIGHT_DOM = 1.0

    


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)


    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print('cfg...')
    print_args(args, cfg)

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    print(trainer.device)

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", default='none', help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", default='all', help="target domains for DA/DG"
    )    
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--strong-transforms", type=list, default=["randaugment_fixmatch"], nargs="+", help="data strong augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
