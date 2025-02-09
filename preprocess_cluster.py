import argparse
import torch
import numpy as np

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import os.path as osp
import os
from dassl.utils import mkdir_if_missing

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
    cfg.DATASET.NUM_SHOTS_UNLABELED = 96   # shots of unlabeled data by default
    cfg.DATASET.CLUSTER_LOADER = 'trainu'     # 'trainu' or 'val' 
    cfg.TRAINER.UNLABELED_CLUSTERS = 3    # the number of clusters for unlabeled data



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
    # import pdb; pdb.set_trace()

    cfg.freeze()

    return cfg



def clustering_indicators(labels_true, labels_pred):
    from sklearn.metrics import f1_score, accuracy_score, normalized_mutual_info_score, rand_score
    from sklearn.preprocessing import LabelEncoder
    if type(labels_true[0]) not in [int, np.int64, np.int32]:
        labels_true = LabelEncoder().fit_transform(labels_true)  
    f_measure = f1_score(labels_true, labels_pred, average='macro')  # F1 score
    accuracy = accuracy_score(labels_true, labels_pred)  # ACC
    normalized_mutual_information = normalized_mutual_info_score(labels_true, labels_pred)  # NMI
    rand_index = rand_score(labels_true, labels_pred)  # RI

    print("F_measure:", f_measure, "ACC:", accuracy, "NMI", normalized_mutual_information, "RI", rand_index)
    return f_measure, accuracy, normalized_mutual_information, rand_index


def knn(data_zs, n_clusters=3):
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=n_clusters, max_iter=500)  
    y_pred = model.fit_predict(data_zs)  
    return y_pred


def domain_cluster(path, vt_fea, n_clusters=3):
    '''preprocessing: cluster unlabeled data'''
    
    v_fea = vt_fea['v_fea']
    dom_label = vt_fea['dom_label']
    domain = vt_fea['domain']
    pred_path = osp.join(path, "vfea_dom_pred_ncluster{}.npy".format(n_clusters))
    
    if not osp.exists(pred_path):
        dom_pred = knn(v_fea, n_clusters)  # KNN
        np.save(pred_path, dom_pred)
    else:
        dom_pred = np.load(pred_path)
    
    clustering_indicators(dom_label, dom_pred)   
    return dom_pred


def write_unlabeled_data_cluster(path, vt_fea, dom_pred, cluster, split='train'):
        domain = vt_fea['domain']
        label = vt_fea['label']
        impath = vt_fea['impath']
        domain_list = set(domain)

        for dom in domain_list:
            filename = dom + "_" + split + f"_cluster{cluster}.txt"
            mkdir_if_missing(osp.join(path, 'splits'))
            split_file = osp.join(path, 'splits', filename)
            dom_idxs = np.nonzero(domain == dom)[0]
            with open(split_file, "w") as f:
                for dom_idx in dom_idxs:
                    line = impath[dom_idx] + ' ' + str(label[dom_idx]) + ' ' + str(dom_pred[dom_idx]) + '\n'
                    f.write(line)
            f.close()


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
    print('trainer2: ',torch.cuda.memory_allocated())

    dataloader = cfg.DATASET.CLUSTER_LOADER
    unlabeled_clusters = cfg.TRAINER.UNLABELED_CLUSTERS

    if unlabeled_clusters:
        data_root = osp.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        dataset_path = str(args.dataset_config_file).split('/')[2][:-5]
        dataset_path = dataset_path.split('_preprocessed')[0]
        dataset_dir = osp.join(data_root, dataset_path)

        if dataloader == 'trainu':
            save_path = osp.join(dataset_dir, 'clusters', f'unlabeled_shots_{cfg.DATASET.NUM_SHOTS_UNLABELED}')
            path = osp.join(save_path, f'unlabeled_clusters{unlabeled_clusters}.npz')
        else:
            save_path = osp.join(dataset_dir, 'clusters', 'test')
            path = osp.join(save_path, f'test_clusters{unlabeled_clusters}.npz')
            mkdir_if_missing(save_path)
    
        
        if not osp.exists(path):
            vt_fea = trainer.get_features_test(dataloader=dataloader)   # dataloader='trainu' or dataloader='val'
            vt_fea['v_fea'] = torch.stack(vt_fea['v_fea']).cpu().numpy()
            vt_fea['t_fea'] = torch.stack(vt_fea['t_fea']).cpu().numpy()
            vt_fea['label'] = torch.stack(vt_fea['label']).cpu().numpy()
            vt_fea['domain'] = np.array(vt_fea['domain'])
            vt_fea['impath'] = np.array(vt_fea['impath'])
            vt_fea['dom_label'] = torch.stack(vt_fea['dom_label']).cpu().numpy()
            np.savez(path, v_fea=vt_fea['v_fea'], t_fea=vt_fea['t_fea'], label=vt_fea['label'], dom_label=vt_fea['dom_label'], domain=vt_fea['domain'], impath=vt_fea['impath'])
        else:
            vt_fea = np.load(path)

        print(f'collect unlabeled features from {path}...')
        dom_pred = domain_cluster(save_path, vt_fea, unlabeled_clusters)

        write_unlabeled_data_cluster(save_path, vt_fea, dom_pred, unlabeled_clusters, split=dataloader)

      

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
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
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
