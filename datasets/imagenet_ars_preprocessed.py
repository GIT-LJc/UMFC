import os
import os.path as osp
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

import random
import copy

TO_BE_IGNORED = ["README.txt", "split_fewshot"]

@DATASET_REGISTRY.register()
class ImageNet_ARS_Preprocessed(DatasetBase):
    """The mixed dataset of ImageNet-Adversarial, ImageNet-Rendition, ImageNet-Sketch-subset. 
        314 classes.
    """

    def __init__(self, cfg):
        dataset_dirs = ["imagenet-a", "imagenet-r", "imagenet-s"]
        self.dataset_dirs = dataset_dirs
        domains = ["adversarial", "rendition", "sketch"]
        self.domain_list = domains
        self.clusters = cfg.TRAINER.UNLABELED_CLUSTERS

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        root = os.path.join(root, "imagenet_ars")
        trains_x =  []
        trains_u =  []
        tests = []
        union_cls = self.get_union_cls(root, domains=["adversarial", "rendition"], dataset_dirs=["imagenet-a", "imagenet-r"])
        print('union classes', union_cls)
        
        tests_sep = dict()
        tests_iodsep = dict()

        
        if cfg.DATASET.TARGET_DOMAINS[0] == 'all':
            unlabeled_domain = domains
        elif ',' in cfg.DATASET.TARGET_DOMAINS[0]:
            unlabeled_domain = cfg.DATASET.TARGET_DOMAINS[0].split(',')
        else:
            unlabeled_domain = cfg.DATASET.TARGET_DOMAINS

        self.seed = cfg.SEED

        
        print('Loading unlabeled data.')
        # root = os.path.join(os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT)), "imagenet_ars_cluster")
        root = os.path.join(os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT)), "imagenet_ars")
        if 'none' in cfg.DATASET.TARGET_DOMAINS:
            trains_u = None
        else:                 # generate data
            for domain in unlabeled_domain: 
                dataset_dir = dataset_dirs[domains.index(domain)]
                self.image_dir = os.path.join(root, "images")
                self.dataset_dir = os.path.join(self.image_dir, dataset_dir)

                self.split_fewshot_dir = os.path.join(root, "clusters")   # use clusters to store the domain clustered few-shot data
                mkdir_if_missing(self.split_fewshot_dir)
                preprocessed = os.path.join(self.split_fewshot_dir, f"{domain}_{self.clusters}-u25_t11-seed_{self.seed}.pkl")
                
                text_file = os.path.join(root, "classnames.txt")
                classnames = self.read_classnames(text_file)

                train, test= self.read_data_split_unlabeled_balanced(classnames=classnames, domain=domain, union_cls=union_cls, cfg=cfg.DATASET.DOMAIN, domains=domains)

                print(domain, 'unlabeled len(train)', len(train))
                print(domain, 'unlabeled len(test)', len(test))

                trains_u.extend(train)
                tests_sep[domain] = test  
                tests.extend(test)

                print(f"Saving preprocessed few-shot data to {preprocessed}")
                data = {"tests": tests, "tests_sep": tests_sep, "train_u": trains_u}
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=3)

            print('overall unlabeled len(trains)', len(trains_u))
            print('overall len(tests)', len(tests))
        
        super().__init__(train_x=None, train_u=trains_u, val=tests, test=tests_sep)



    def get_union_cls(self, root, domains, dataset_dirs):
        union_cls = set()
        classnames_register = dict()
         
        for domain, dataset_dir in zip(domains, dataset_dirs): 
            dataset_dir = os.path.join(root, 'images', dataset_dir)
            # dataset_dir = os.path.join(root, dataset_dir)
            folders = listdir_nohidden(dataset_dir, sort=True)
            folders = [f for f in folders if f not in TO_BE_IGNORED]
            print(dataset_dir)
            text_file = os.path.join(root, "classnames.txt")
            classnames = self.read_classnames(text_file)
            classname_regis = [classnames[folder] for folder in folders]
            union_cls = union_cls | set(classname_regis)

        union_cls = list(union_cls)
        union_cls.sort()

        return union_cls

    # For initial data sampling and preprocessing
    def read_data_split_unlabeled_balanced(self, classnames, union_cls, domain='' ,split=0.7, cfg=0, domains='', label_cls=None):
        image_dir = self.dataset_dir
        folders = listdir_nohidden(image_dir, sort=True)
        folders = [f for f in folders if f not in TO_BE_IGNORED]

        train_items = []
        test_items = []
        n_cls = len(union_cls)
        dom_label = domains.index(domain) # ground-truth domain label 

        random.seed(0)
        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            
            if classname in union_cls:
                if label_cls is not None:
                    if classname in label_cls:
                        label_register = label_cls.index(classname)
                    else:
                        label_register = len(label_cls)
                        classname = 'unknown' 
                else:
                    label_register = union_cls.index(classname)
            else:
                continue
            
            sizes = len(imnames)

            train_n = 25
            train_all = int(sizes*split)
            train_n = min(train_all, train_n)

            test_n = 11
            test_all = sizes - train_all
            test_n = min(test_all, test_n)        

            train_idx = random.sample(range(0,train_all), train_n)
            test_idx = random.sample(range(train_all,sizes), test_n)

            for idx in train_idx:
                imname = imnames[idx]
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label_register, classname=classname, domain=domain, dom_label=dom_label)
                train_items.append(item)
  
            for idx in test_idx:
                imname = imnames[idx]
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label_register, classname=classname, domain=domain, dom_label=dom_label)
                test_items.append(item)
        return train_items, test_items


    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = dict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames