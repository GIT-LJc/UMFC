import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights

from tqdm import tqdm
from collections import defaultdict
import numpy as np  
import os
from dassl.utils import mkdir_if_missing

from .zsclip import load_clip_to_cpu
from .umfc_UC import UMFC

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
class prototype_cluster():
    def __init__(self, n_clusters):
        self.model = KMeans(n_clusters=n_clusters, max_iter=500)  
        self.centroids = None
        self.counter = [0] * n_clusters
        self.label = list(range(n_clusters))

    def start(self, x):
        x = x.cpu()
        y_pred = self.model.fit_predict(x)
        self.centroids = self.model.cluster_centers_
        self.count(y_pred)
        return y_pred

    def pred_update(self, batchx):
        batchx = batchx.cpu().numpy()
        y_pred = pairwise_distances_argmin(batchx, self.centroids)
        for i in self.label:
            self.centroids[i] = (self.centroids[i] * self.counter[i] + sum(batchx[y_pred == i])) / (self.counter[i] + sum(y_pred == i))
        self.count(y_pred)
        return y_pred
    
    def count(self, y_pred):
        for i in self.label:
            self.counter[i] += sum(y_pred == i)


@TRAINER_REGISTRY.register()
# 即 ZeroshotCLIP_TTC_prototype
class UMFC_TTC(UMFC):
    '''test-time calibration'''

    def build_model(self):
        cfg = self.cfg
        self.lab2cname = self.dm.dataset.lab2cname
        classnames = self.dm.dataset.classnames
        self.classnames = classnames
        self.remove_classes = []
        self.superclass = []

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)
        
        prompts = [self.temp.format(c.replace("_", " ")) for c in self.classnames]

        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        
        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)

        self.text_features = text_features
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.T = 1.0
        self.conf_thre = 0.95
        self.alpha = cfg.TRAINER.CALIBRATE_IMG_WEIGHT
        self.beta = cfg.TRAINER.CALIBRATE_TEXT_WEIGHT
        self.gamma = cfg.TRAINER.EMA_DECAY
        self.keys = list(range(self.cfg.TRAINER.UNLABELED_CLUSTERS))
        self.domains = self.dm.dataset.domain_list
        
        self.calibrate_init(cfg, classnames) # init the calibration information
      

    def calibrate_init(self, cfg, classnames):
        self.image_bias_logits = torch.zeros(len(classnames)).to('cuda')
        self.text_bias_logits = torch.zeros(len(classnames)).to('cuda')
        self.image_bias_features = 0
        self.text_bias_features = 0

        self.domimg_bias_logits = {}
        self.domimg_bias_features = {}
        self.domtext_bias_logits = {}
        self.domtext_bias_features = {}

        self.domain_img = defaultdict(list)
        self.domain_feat = defaultdict(list)
        self.domain_logit = defaultdict(list)
        self.domain_prob = defaultdict(list)
        
        self.domain_img_avg = defaultdict(float)
        self.domain_feat_avg = defaultdict(float)
        self.domain_logit_avg = defaultdict(float)
        self.domain_prob_avg = defaultdict(float)

        self.main_domain_list = defaultdict(list)
        self.dim = self.text_features.shape[1]

        for i in range(self.cfg.TRAINER.UNLABELED_CLUSTERS):
            self.domimg_bias_logits[i] = torch.zeros(1, len(classnames), dtype=self.dtype).to('cuda')
            self.domimg_bias_features[i] = torch.zeros(1, self.dim, dtype=self.dtype).to('cuda')
            self.domtext_bias_logits[i] = torch.zeros(1, len(classnames), dtype=self.dtype).to('cuda')
            self.domtext_bias_features[i] = torch.zeros(1, self.dim, dtype=self.dtype).to('cuda')


    def pred_domain(self, image):
        domlabel = self.pred_model(image)
        self.pred_model.update()    
        return domlabel

    # calibrate名字中含‘img’, batch update
    @torch.no_grad()
    def get_image_bias(self, batch):
        domain_img_batch = defaultdict(list)
        domain_feat_batch = defaultdict(list)
        domain_logit_batch = defaultdict(list)

        self.set_model_mode('eval')
    
        parsed_data = self.parse_batch_train_dompred(batch)
        input_u, index_u, label_u, domlabel_u, domname_list_u = parsed_data

        outputs_u, img_feat = self.get_logits_features(input_u)
        probs_u = torch.softmax(outputs_u, dim=-1)
        max_probs, targets_u = torch.max(probs_u, dim=-1)
        mask = max_probs.ge(self.conf_thre).int()
        for i, name in enumerate(domlabel_u):      # domlabel_u is a list of domain pred (clustered domain)
            self.main_domain_list[name].append(mask[i])
            # image calibration
            domain_logit_batch[name].append(outputs_u[i].cpu())
            # text calibration
            domain_feat_batch[name].append(img_feat[i].cpu())
        
        return domain_img_batch, domain_feat_batch, domain_logit_batch

    # update bias in each batch
    def update_img_bias(self, batch_information):   
        self.main_domain = sorted(self.main_domain_list.keys(), key=lambda item: sum(self.main_domain_list[item])/(len(self.main_domain_list[item]) + 1e-5), reverse=True)[0]    
        domain_img_batch, domain_feat_batch, domain_logit_batch =  batch_information   # current information
        update_keys = self.get_keys(batch_information)
        
        # ttc_update type
        if self.cfg.TTC_UPDATE == 'memory':
            # image calibration
            self.domain_logit_avg = self.dict_add_list(self.domain_logit, domain_logit_batch, self.domain_logit_avg)
            # text calibration
            self.domain_feat_avg = self.dict_add_list(self.domain_feat, domain_feat_batch, self.domain_feat_avg)
        elif self.cfg.TTC_UPDATE == 'ema':
            # image calibration
            self.domain_logit_avg = self.dict_update_list(self.domain_logit_avg, domain_logit_batch)
            # text calibration
            self.domain_feat_avg = self.dict_update_list(self.domain_feat_avg, domain_feat_batch)

        for name in update_keys:
            # image calibration
            self.domimg_bias_logits[name] = self.domain_logit_avg[name].cuda()
            
            # text calibration
            domain_feat_avg = - self.domain_feat_avg[name] + self.domain_feat_avg['avg']
            self.domtext_bias_features[name] = domain_feat_avg.cuda()  
            if 'ensemble' in self.cfg.TRAINER.CALIBRATE_TEXT:
                self.text_bias_features = sum(self.domtext_bias_features.values()).cuda()


    def get_keys(self, dicts):
        keys = []
        for d in dicts:
            keys = set(keys).union(set(d.keys()))
        return list(keys)


    def dict_add_list(self, dica, dicb, dicc):
        for k in self.keys:
            if k in dicb.keys():
                dica[k].extend(dicb[k])
                dicc[k] = self.calculate_large_tensor(dica[k]).unsqueeze(dim=0)
        tmp = torch.cat([torch.stack(dica[name]) for name in dica.keys()], dim=0)
        dicc['avg'] = self.calculate_large_tensor(tmp).unsqueeze(dim=0)
        return dicc
    

    def dict_update_list(self, dica, dicb, batchid=4):
        # gamma = batchid / (batchid + 1)
        for k in dicb.keys():
            avg_b = torch.stack(dicb[k], dim=0).mean(dim=0).unsqueeze(dim=0)
            dica[k] = self.gamma * dica[k] + (1 - self.gamma) * avg_b
        dica['avg'] = torch.cat([dica[name] for name in dica.keys()], dim=0).mean(dim=0, keepdim=True)
        return dica

    def domain_evaluator_process(self, output, label):
        keys = list(set(self.domain))
        for d in keys:
            idx = [i for i,x in enumerate(self.domain) if x==d]
            out = output[idx]
            lab = label[idx]
            self.evaluator_dict[d].process(out, lab)

    def domain_evaluator_evaluate(self):
        accs = {}
        keys = self.domains
        for d in keys:
            print('Evaluate on domain:', d)
            res = self.evaluator_dict[d].evaluate()
            accs[d] = float(format(res['accuracy'], '.2f'))
        return accs


    def get_image_features(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def parse_batch_train_dompred(self, batch_u):
        input_u = batch_u["img"]  # weak augmentation
        index_u = batch_u['index']
        label_u = batch_u["label"]
        domname_list_u = batch_u["domain"]

        input_u = input_u.to(self.device)
        index_u = index_u.to(self.device)
        label_u = label_u.to(self.device)
        domlabel_u = self.domlabel
        # domlabel_u = self.domlabel.numpy()

        return input_u, index_u, label_u, domlabel_u, domname_list_u
    

    @torch.no_grad()
    def test(self, split='val'):
        
        """A generic test-time calibration pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        import copy
        cur_evaluator = copy.deepcopy(self.evaluator)

        self.evaluator_dict = {}   # evaluate per domain
        # for key in self.keys:
        for key in self.domains:
            self.evaluator_dict[key] = copy.deepcopy(self.evaluator)
            self.evaluator_dict[key].reset()   

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val":
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        len_dom = 0
           
        print(f'Test-Time Calibration Accuracy' )  

        import time
        start_time = time.time()
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            domain = batch['domain']    # gt label, for test-acc 
            self.domain = domain

            img_feature = self.get_image_features(input)

            if batch_idx == 0:      # only for the first batch
                self.pred_model = prototype_cluster(n_clusters=self.cfg.TRAINER.UNLABELED_CLUSTERS)    # initialize the prototype cluster model
                domlabel = self.pred_model.start(img_feature)
            else:
                domlabel = self.pred_model.pred_update(img_feature)    # predicted domain label

            self.domlabel = domlabel      
            batch_information = self.get_image_bias(batch)    # collect image bias information
            self.update_img_bias(batch_information)    # update image bias information
            output = self.model_inference(input, domlabel)
            
            self.evaluator.process(output, label, len_dom)
            self.domain_evaluator_process(output, label)
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx+1} batches")

        results = self.evaluator.evaluate()
        accuracys = self.domain_evaluator_evaluate()
        average = float(format(sum(accuracys.values())/len(accuracys.values()), '.2f'))
        print(
            "=> summary results \n"
            f"{accuracys} \n"
            f"average results: {average:.2f}%"
        )

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

        return 0




@TRAINER_REGISTRY.register()
class UMFC_TTC_Summary(UMFC_TTC):
 

    @torch.no_grad()
    def test(self, split='val'):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        import copy
        cur_evaluator = copy.deepcopy(self.evaluator)
        
        self.evaluator_dict = {}   # evaluate per domain
        for key in self.domains:
            self.evaluator_dict[key] = copy.deepcopy(self.evaluator)
            self.evaluator_dict[key].reset() 

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val":
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        len_dom = 0
        
        summary_accuracy = defaultdict(list)
        alpha_list = np.arange(0.0, 1.2, 0.1)
        beta_list = np.arange(0.0, 1.0, 0.1)
        
        for alpha in alpha_list:
            self.alpha = alpha
            for beta in beta_list:
                self.beta = beta
                
                self.calibrate_init(self.cfg, self.classnames)  # initialize the calibrate bias

                for batch_idx, batch in enumerate(tqdm(data_loader)):
                    input, label = self.parse_batch_test(batch)
                    domain = batch['domain']    # gt label, for test-acc 
                    self.domain = domain

                    img_feature = self.get_image_features(input)

                    if batch_idx == 0:      # only for the first batch
                        self.pred_model = prototype_cluster(n_clusters=self.cfg.TRAINER.UNLABELED_CLUSTERS)    # initialize the prototype cluster model
                        domlabel = self.pred_model.start(img_feature)
                    else:
                        domlabel = self.pred_model.pred_update(img_feature)    # predicted domain label

                    self.domlabel = domlabel

                    batch_information = self.get_image_bias(batch)    # collect image bias information

                    self.update_img_bias(batch_information)    # update image bias information

                    output = self.model_inference(input, domlabel)
              
                    self.evaluator.process(output, label, len_dom)
                    self.domain_evaluator_process(output, label)
                    if batch_idx % 10 == 0:
                        print(f"Processed {batch_idx+1} batches")

                results = self.evaluator.evaluate()
                accuracys = self.domain_evaluator_evaluate()
                average = float(format(sum(accuracys.values())/len(accuracys.values()), '.2f'))
                print(
                    f"=> summary results alpha{alpha:.2f} - beta{beta:.2f}\n"
                    f"{accuracys} \n"
                    f"average results: {average:.2f}%"
                )
                summary_accuracy = self.summary_domain(summary_accuracy, accuracys)
                summary_accuracy['average'].append(average)
                
                
        for k, v in summary_accuracy.items():
            print(
                f"=> summary results on {k}:\n"
                f"{v} \n"
                f"Best accuracy: {max(summary_accuracy[k])}\n"
            )
        
        print('=================>>> reshape the results <<<=================')
        print('alpha_list: ', alpha_list)
        print('beta_list: ', beta_list)
        for k, v in summary_accuracy.items():
            reshape_v = np.array(v).reshape(len(alpha_list), len(beta_list)).tolist()
            print(f"=> summary results on {k}:")
            for rv in reshape_v:
                print(rv)
            print(f"Best accuracy: {max(summary_accuracy[k])}\n")
            
        print('Best accuracy: ', max(summary_accuracy['average']))

        return 0   

    def summary_domain(self, dica, dicb):
        for k, v in dicb.items():
            dica[k].append(v)
        return dica

