import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights

from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT, DOMAINNET_TEMPLATES

from tqdm import tqdm
from collections import defaultdict
import numpy as np  
import os
from dassl.utils import mkdir_if_missing

from trainers.zsclip import ZeroshotCLIP


class prototype_updater():
    def __init__(self, text_features, filter_k):
        self.centroids = copy.deepcopy(text_features)
        self.filter_k = filter_k if filter_k>0 else 200
        self.num_classes = text_features.shape[0]
        self.counter = [1] * self.num_classes
        self.label = list(range(self.num_classes))
    
    def count(self, y_pred):
        for i in self.label:
            self.counter[i] += sum(y_pred == i)

    def update(self, img_features, ent, targets_u):
        indices = []
        indices1 = torch.LongTensor(list(range(len(ent)))).cuda()  
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent[targets_u == i])
            indices_filtered = indices1[targets_u==i][indices2][:self.filter_k]
            self.centroids[i] = (self.centroids[i] * self.counter[i] + sum(img_features[indices_filtered])) / (self.counter[i] + len(indices_filtered))
            indices.append(indices1[targets_u==i][indices2][:self.filter_k])
        indices = torch.cat(indices)
        targets_u_filtered = targets_u[indices]
        self.count(targets_u_filtered)
    
    def reset(self, text_features, filter_k):
        self.centroids = copy.deepcopy(text_features)
        self.filter_k = filter_k if filter_k>0 else 200
        self.num_classes = text_features.shape[0]
        self.counter = [1] * self.num_classes
        self.label = list(range(self.num_classes))


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

import copy
@TRAINER_REGISTRY.register()
class ZeroshotCLIP_T3A_prototype(ZeroshotCLIP):
    '''test-time calibration with prototype clustering'''
    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_classes = len(self.classnames)
        self.domains = self.dm.dataset.domain_list
        self.T = 1.0

        self.filter_k = cfg.T3A_FilterK

        self.prototype_updater = prototype_updater(self.text_features, self.filter_k)
        self.class_prototypes = self.prototype_updater.centroids
        self.weight_t3a = 1.0
        

    def model_inference(self, image, logits_clip=0):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits_prot = logit_scale * image_features @ self.class_prototypes.t()
        logits = self.weight_t3a * logits_prot + (1-self.weight_t3a) * logits_clip
        return logits

    def clip_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits_clip = logit_scale * image_features @ self.text_features.t()
        pseudo_label = torch.softmax(logits_clip.detach() / self.T, dim=-1)
        ent = softmax_entropy(pseudo_label)
        max_probs, targets_u  = torch.max(pseudo_label, dim=-1)
        return image_features, logits_clip, ent, targets_u




    def get_image_features(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def parse_batch_train_dompred(self, batch_u):
        input_u = batch_u["img"]  
        index_u = batch_u['index']
        label_u = batch_u["label"]
        domname_list_u = batch_u["domain"]

        input_u = input_u.to(self.device)
        index_u = index_u.to(self.device)
        label_u = label_u.to(self.device)
        domlabel_u = self.domlabel

        return input_u, index_u, label_u, domlabel_u, domname_list_u
    

    @torch.no_grad()
    def test(self, split='val'):
        
        """A generic test-time calibration pipeline."""
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
        print(f'Test-Time Calibration Accuracy' )  

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            domain = batch['domain']    # gt label, for test-acc 
            self.domain = domain

            image_features, logits_clip, ent, targets_u = self.clip_inference(input)
            self.prototype_updater.update(image_features, ent, targets_u)
            self.class_prototypes = self.prototype_updater.centroids

            output = self.model_inference(input, logits_clip)
            
            self.evaluator.process(output, label, len_dom)
            self.domain_evaluator_process(output, label)
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx+1} batches")
                results = self.evaluator.evaluate()
                results = self.domain_evaluator_evaluate()

        results = self.evaluator.evaluate()
        accuracys = self.domain_evaluator_evaluate()
        average = float(format(sum(accuracys.values())/len(accuracys.values()), '.2f'))
        print(
            "=> summary results \n"
            f"{accuracys} \n"
            f"average results: {average:.2f}%"
        )

        return 0


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



@TRAINER_REGISTRY.register()
class ZeroshotCLIP_T3A_prototype_Summary(ZeroshotCLIP_T3A_prototype):
 
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
        alpha_list = [1, 5, 20, 50, 100]    # the filterk
        beta_list = np.arange(0.5, 1.1, 0.1)    # the weight used in t3a 
        for alpha in alpha_list:
            self.filter_k = alpha
            for beta in beta_list:
                self.weight_t3a = beta
                #! initialize the prototype bias
                self.prototype_updater.reset(self.text_features, self.filter_k)
                self.class_prototypes = self.prototype_updater.centroids
                
                for batch_idx, batch in enumerate(tqdm(data_loader)):
                    input, label = self.parse_batch_test(batch)
                    domain = batch['domain']    # gt label, for test-acc 
                    self.domain = domain
                    
                    image_features, logits_clip, ent, targets_u = self.clip_inference(input)
                    self.prototype_updater.update(image_features, ent, targets_u)
                    self.class_prototypes = self.prototype_updater.centroids

                    output = self.model_inference(input, logits_clip)

                    self.evaluator.process(output, label, len_dom)
                    self.domain_evaluator_process(output, label)
                    if batch_idx % 10 == 0:
                        print(f"Processed {batch_idx+1} batches")
                        results = self.evaluator.evaluate()
                        # results = self.domain_evaluator_evaluate()

                results = self.evaluator.evaluate()
                accuracys = self.domain_evaluator_evaluate()
                average = float(format(sum(accuracys.values())/len(accuracys.values()), '.2f'))
                print(
                    f"=> summary results filterk{alpha:.2f} - weightt3a{beta:.2f}\n"
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
        print('filterk_list: ', alpha_list)
        print('weightt3a_list: ', beta_list)
        for k, v in summary_accuracy.items():
            reshape_v = np.array(v).reshape(len(alpha_list), len(beta_list)).tolist()
            print(f"=> summary results on {k}:")
            for rv in reshape_v:
                print(rv)
            print(f"Best accuracy: {max(summary_accuracy[k])}\n")
            
        print('Best accuracy: ', max(summary_accuracy['average']))

        return 0   # 仅涉及测试阶段，无需保存current result

    def summary_domain(self, dica, dicb):
        for k, v in dicb.items():
            dica[k].append(v)
        return dica

