import os
import os.path as osp
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing


TO_BE_IGNORED = ["README.txt", "split_fewshot"]

@DATASET_REGISTRY.register()
class ImageNet_ARS_Cluster(DatasetBase):
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
        image_dir = os.path.join(root, "images")
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
        elif os.path.exists(root):    # load data
            trains_u, tests, tests_sep = self.get_dataset_unlabeled_cluster(unlabeled_domain, root)
        else:
            raise NotImplementedError(f'Please preprocess to obtain clustering labels first.')
        
        tests = self.shuffle_list(tests)

        super().__init__(train_x=None, train_u=trains_u, val=tests, test=tests_sep)

    def shuffle_list(self, data_list):
        import random
        random.seed(self.seed)
        random.shuffle(data_list)
        return data_list

    def get_union_cls(self, root, domains, dataset_dirs):
        union_cls = set()
        classnames_register = dict()
         
        for domain, dataset_dir in zip(domains, dataset_dirs): 
            dataset_dir = os.path.join(root, "images", dataset_dir)
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



    def _read_unlabeled_data_cluster(self, domain, split="train"):
        items = []

        filename = domain + "_" + split + f"_cluster{self.clusters}.txt"
        split_file = osp.join(self.cluster_dir, 'splits', filename)
        print('split_file:', split_file)

        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if '.jpg' in line:
                    impath, ld = line.split(".jpg")
                    impath = impath + ".jpg"
                elif '.JPEG' in line:
                    impath, ld = line.split(".JPEG")
                    impath = impath + ".JPEG"
                elif '.png' in line:
                    impath, ld = line.split(".png")
                    impath = impath + ".png"
                else:
                    print('error')
                ld = ld.strip()
                splits = ld.split(" ")
                if len(splits) == 3:
                    label, dom_pred, classname = splits
                else:
                    label = splits[0]
                    dom_pred = splits[1]
                    classname = ' '.join(splits[2:])
                
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


    def get_dataset_unlabeled_cluster(self, domains, root):
        trains_u =  []
        tests = []
        tests_sep = dict()
        for domain in domains: 
            dataset_dir = self.dataset_dirs[domains.index(domain)]
            self.dataset_dir = os.path.join(root, dataset_dir)
            self.split_fewshot_dir = os.path.join(root, "clusters")   # use clusters to store the domain clustered data
            mkdir_if_missing(self.split_fewshot_dir)
            preprocessed = os.path.join(self.split_fewshot_dir, f"{domain}_{self.clusters}-u25_t11-seed_{self.seed}.pkl")
            if os.path.exists(preprocessed):  
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    trains_u = data["train_u"]
                    tests = data["tests"]
                    tests_sep = data["tests_sep"]
            else:
                self.cluster_dir = os.path.join(self.split_fewshot_dir, 'unlabeled_shots_96')
                train = self._read_unlabeled_data_cluster(domain, split="train")
                self.cluster_dir = os.path.join(self.split_fewshot_dir, 'test')
                test = self._read_unlabeled_data_cluster(domain, split="test")

                print(domain, 'unlabeled len(train)', len(train))
                print(domain, 'unlabeled len(test)', len(test))

                trains_u.extend(train)
                tests_sep[domain] = test   
                tests.extend(test) 
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                data = {"tests": tests, "tests_sep": tests_sep, "train_u": trains_u}
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=3)
      
                print(domain, 'len(train):', len(train))
                
        print('overall unlabeled len(trains)', len(trains_u))
        print('overall len(tests)', len(tests))
        return trains_u, tests, tests_sep

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

