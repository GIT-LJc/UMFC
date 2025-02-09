import os
import os.path as osp
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

TO_BE_IGNORED = ["README.txt"]

@DATASET_REGISTRY.register()
class DomainNet_Cluster(DatasetBase):

    def __init__(self, cfg):
        domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        self.domain_list = domains
        self.domain2domlabel = {'clipart':0, 'infograph':1, 'painting':2, 'quickdraw':3, 'real':4, 'sketch':5}

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, "domainnet")
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
        
        # self.cluster_dir = os.path.join(self.dataset_dir, "clusters", "test_prototype")
        self.cluster_dir = os.path.join(self.dataset_dir, "clusters", "test")
        tests_sep, tests = self.get_test_dataset_cluster(domains)   
    
        DatasetBase.__init__(self, train_x=None, train_u=trains_u, val=tests, test=tests_sep)



    def get_test_dataset_cluster(self, domains):
        tests = []
        tests_sep = dict()
        for domain in domains: 
            test = self._read_unlabeled_data_cluster(domain, split="test")    
            preprocessed = os.path.join(self.dataset_dir, "split_fewshot", domain, f"test_balanced_cluster{self.clusters}_prototype.pkl")
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot test data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    test = data["test"]
            else:
                print(f"Saving preprocessed few-shot test data to {preprocessed}")
                test = self.generate_fewshot_dataset(test, num_shots=30)
                data = {"test": test}
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=3)
                    
            print(domain, 'len(test):', len(test))

            
            tests_sep[domain] = test   # tests 分别测试
            tests.extend(test)
        print('overall len(tests)', len(tests))
        return tests_sep, tests




    def get_dataset_unlabeled_cluster(self, domains, num_shots):
        trains =  []
        for domain in domains: 
            self.split_fewshot_dir = os.path.join(self.cluster_dir, "split_fewshot")
            mkdir_if_missing(self.split_fewshot_dir)
            mkdir_if_missing(os.path.join(self.split_fewshot_dir, domain))
        
            self.split_dir = osp.join(self.dataset_dir, "splits")
            train = self._read_unlabeled_data_cluster(domain, split="train")

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


    def _read_unlabeled_data_cluster(self, domain, split="train"):
        items = []

        filename = domain + "_" + split + f"_cluster{self.clusters}.txt"
        split_file = osp.join(self.cluster_dir, 'splits', filename)

        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                impath, label, dom_pred = line.split(" ")
                relpath = self.str_sub(impath, self.image_dir)
                classname = relpath.split("/")[2]
                impath = osp.join(self.image_dir, impath)
                label = int(label)
                dom_pred = int(dom_pred)

                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    dom_label=dom_pred,
                    classname=classname
                )
                items.append(item)

        return items


    def str_sub(self, str1, str2):
        if str1.find(str2) != -1:
            return str1.replace(str2, '')

