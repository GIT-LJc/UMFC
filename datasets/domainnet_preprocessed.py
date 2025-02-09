# for preprocess
import os
import os.path as osp
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

import random
import copy

TO_BE_IGNORED = ["README.txt"]

@DATASET_REGISTRY.register()
class DomainNet_Preprocessed(DatasetBase):
    """
        DomainNet
    """
    dataset_dir = "domainnet"

    def __init__(self, cfg):
        domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        self.domain_list = domains
        self.domain2domlabel = {'clipart':0, 'infograph':1, 'painting':2, 'quickdraw':3, 'real':4, 'sketch':5}

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")

        trains_x =  []
        trains_u =  []
        tests = []
        tests_sep = dict()

        if cfg.DATASET.TARGET_DOMAINS[0] == 'all' or (cfg.DATASET.TARGET_DOMAINS[0] == 'none'):
            unlabeled_domain = domains
        elif ',' in cfg.DATASET.TARGET_DOMAINS[0]:
            unlabeled_domain = cfg.DATASET.TARGET_DOMAINS[0].split(',')
        else:
            unlabeled_domain = cfg.DATASET.TARGET_DOMAINS
        unlabeled_num_shots = int(cfg.DATASET.NUM_SHOTS_UNLABELED / len(unlabeled_domain))   

        self.seed = cfg.SEED

        print('Loading unlabeled data.')
        trains_u = self.get_train_dataset(unlabeled_domain, unlabeled_num_shots)
        print('Loading test data.')
        tests_sep, tests = self.get_test_dataset(domains)

        super().__init__(train_x=None, train_u=trains_u, val=tests, test=tests_sep)
       

    def get_train_dataset(self, domains, num_shots):
        trains =  []
        for _, domain in enumerate(domains): 
            self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
            mkdir_if_missing(self.split_fewshot_dir)
            mkdir_if_missing(os.path.join(self.split_fewshot_dir, domain))
            
            self.split_dir = osp.join(self.dataset_dir, "splits")

            train = self._read_data(domain, split="train")

            if num_shots >= 1:
                seed = self.seed 
                preprocessed = os.path.join(self.split_fewshot_dir, domain, f"shot_{num_shots}-seed_{seed}.pkl")
            
                if os.path.exists(preprocessed):
                    print(f"Loading preprocessed few-shot data from {preprocessed}")
                    with open(preprocessed, "rb") as file:
                        data = pickle.load(file)
                        train = data["train"]
                else:
                    train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                    data = {"train": train}
                    print(f"Saving preprocessed few-shot data to {preprocessed}")
                    with open(preprocessed, "wb") as file:
                        pickle.dump(data, file, protocol=3)


            print(domain, 'len(train):', len(train))

            trains.extend(train)
        print('overall len(trains)', len(trains))
        return trains


    def get_test_dataset(self, domains):
        tests = []
        tests_sep = dict()
        for dlabel, domain in enumerate(domains): 
            test = self._read_data(domain, split="test")   
            preprocessed = os.path.join(self.dataset_dir, "split_fewshot", domain, f"test_balanced.pkl")
            if os.path.exists(preprocessed):
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    test = data["test"]
            else:
                test = self.generate_fewshot_dataset(test, num_shots=30)
                data = {"test": test}
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=3)
            print(domain, 'len(test):', len(test))
            tests_sep[domain] = test   # tests 分别测试
            tests.extend(test)
        print('overall len(tests)', len(tests))
        return tests_sep, tests



    def _read_data(self, domain, split="train"):
        items = []

        filename = domain + "_" + split + ".txt"
        split_file = osp.join(self.split_dir, filename)
        dom_label = self.domain2domlabel[domain]

        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                classname = impath.split("/")[1]
                impath = osp.join(self.image_dir, impath)
                label = int(label)
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    dom_label=dom_label,
                    classname=classname
                )
                items.append(item)

        return items


