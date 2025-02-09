# For Test-Time Adaptation
import os
import os.path as osp
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

import random

TO_BE_IGNORED = ["README.txt"]
from .domainnet_cluster import DomainNet_Cluster

# DomainNet_Cluster_Prototype_TTC 2.1
@DATASET_REGISTRY.register()
class DomainNet_Cluster_TTC(DomainNet_Cluster):
    """
        DomainNet
    """
    dataset_dir = "domainnet"
    clean = False

    def __init__(self, cfg):
        domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        self.domain_list = domains
        self.domain2domlabel = {'clipart':0, 'infograph':1, 'painting':2, 'quickdraw':3, 'real':4, 'sketch':5}

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        
        self.clusters = cfg.TRAINER.UNLABELED_CLUSTERS
        self.cluster_dir = os.path.join(self.dataset_dir, "clusters", f"unlabeled_shots_{cfg.DATASET.NUM_SHOTS_UNLABELED}")
       

        trains_u =  []
        tests = []
        tests_sep = dict()


        if cfg.DATASET.TARGET_DOMAINS[0] == 'all':
            unlabeled_domain = domains
        elif ',' in cfg.DATASET.TARGET_DOMAINS[0]:
            unlabeled_domain = cfg.DATASET.TARGET_DOMAINS[0].split(',')
        else:
            unlabeled_domain = cfg.DATASET.TARGET_DOMAINS
        unlabeled_num_shots = int(cfg.DATASET.NUM_SHOTS_UNLABELED / len(unlabeled_domain))   

        self.seed = cfg.SEED


        if 'none' in cfg.DATASET.TARGET_DOMAINS:
            trains_u = None
        else:
            trains_u = self.get_dataset_unlabeled_cluster(unlabeled_domain, unlabeled_num_shots)

        self.cluster_dir = os.path.join(self.dataset_dir, "clusters", "test")
        tests_sep, tests = self.get_test_dataset_cluster(domains)   

        tests = self.shuffle_list(tests)

        DatasetBase.__init__(self, train_x=None, train_u=trains_u, val=tests, test=tests_sep)

    def shuffle_list(self, data_list):
        random.seed(self.seed)
        random.shuffle(data_list)
        return data_list
