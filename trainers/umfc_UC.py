import torch
import torch.nn as nn
import torch.nn.functional as F

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
from .zsclip import load_clip_to_cpu, ZeroshotCLIP


@TRAINER_REGISTRY.register()
class UMFC(ZeroshotCLIP):
    """Domain Prompt."""
    temp = "a photo of a {}."

    templates = {
        "clipart": "a clipart image of a {}.",
        "infograph": "an infograph image of a {}.",
        "painting": "a painting image of a {}.",
        "quickdraw": "a quickdraw image of a {}.",
        "real": "a real image of a {}.",
        "sketch": "a sketch image of a {}.",
        }
    

    def build_model(self):
        cfg = self.cfg
        self.lab2cname = self.dm.dataset.lab2cname
        classnames = self.dm.dataset.classnames
        self.classnames = classnames
        self.domains = self.dm.dataset.domain_list
        self.remove_classes = []
        self.superclass = []
        self.eval_train_loader_u = self.dm.eval_train_loader_u


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
        self.calibrate(cfg, classnames)
        

    def forward(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        return logits


    def get_logits_features(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        return logits, image_features

    def model_inference(self, image, domain=None):
        if domain is not None:
            dom_list = [d.item() for d in domain]

            self.image_bias_logits = torch.stack([self.domimg_bias_logits[d] for d in dom_list]).squeeze()
            self.text_bias_logits = torch.stack([self.domtext_bias_logits[d] for d in dom_list]).squeeze()
            self.image_bias_features = torch.stack([self.domimg_bias_features[d] for d in dom_list]).squeeze(dim=1)
            if 'ensemble' not in self.cfg.TRAINER.CALIBRATE_TEXT:    
                self.text_bias_features = torch.stack([self.domtext_bias_features[d] for d in dom_list]).squeeze(dim=1)
        
        logits = self.calibrate_logits(image)
        return logits
    

    def calibrate_logits(self, image):
        image_features = self.clip_model.encode_image(image)
        text_features = self.text_features

        # calibrate image features
        ca_img_features = image_features - self.alpha * self.image_bias_features
        ca_img_features = ca_img_features / ca_img_features.norm(dim=-1, keepdim=True)

        # calibrate text features
        if 'norm' in self.cfg.TRAINER.CALIBRATE_TEXT:
            text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
            text_bias_features = self.text_bias_features / self.text_bias_features.norm(dim=-1, keepdim=True)
        else:
            text_features = self.text_features
            text_bias_features = self.text_bias_features
        ca_text_features = text_features - self.beta * text_bias_features
        ca_text_features = ca_text_features / ca_text_features.norm(dim=-1, keepdim=True)

        ca_logits = self.clip_model.logit_scale.exp() * ca_img_features @ ca_text_features.t()

        # calibrate logits
        ca_logits = ca_logits - self.alpha * self.image_bias_logits - self.beta * self.text_bias_logits

        return ca_logits


    def calibrate(self, cfg, classnames):
        with torch.no_grad():
            # initialize bias
            self.image_bias_logits = torch.zeros(len(classnames)).to('cuda')
            self.text_bias_logits = torch.zeros(len(classnames)).to('cuda')
            self.image_bias_features = 0
            self.text_bias_features = 0

            self.domimg_bias_logits = {}
            self.domimg_bias_features = {}
            self.domtext_bias_logits = {}
            self.domtext_bias_features = {}

            for i in range(self.cfg.TRAINER.UNLABELED_CLUSTERS):
                self.domimg_bias_logits[i] = torch.zeros(1,len(classnames), dtype=self.dtype).to('cuda')
                self.domimg_bias_features[i] = torch.tensor([[0]], dtype=self.dtype).to('cuda')
                self.domtext_bias_logits[i] = torch.zeros(len(classnames), dtype=self.dtype).to('cuda')
                self.domtext_bias_features[i] = torch.tensor([[0]], dtype=self.dtype).to('cuda')

            self.get_image_bias()

            print('calibration done')
            # self.draw_bias_dom(classnames)

            


    @torch.no_grad()
    def get_image_bias(self):
        domain_feat_avg = defaultdict(list)
        domain_logit_avg = defaultdict(list)
        self.domain_img_avg = {}
        self.domain_feat_avg = {}
        self.domain_logit_avg = {}
        self.domain_prob_avg = {}
        self.main_domain = defaultdict(int)
        from tqdm import tqdm
        loader = self.val_loader
        self.set_model_mode('eval')
        keys = []

        for batch_idx, batch in enumerate(tqdm(loader, desc="Collecting domain average information")):
            parsed_data = self.parse_batch_train_dompred(batch)
            input_u, index_u, label_u, domlabel_u, domname_list_u = parsed_data

            outputs_u, img_feat = self.get_logits_features(input_u)
            probs_u = torch.softmax(outputs_u, dim=-1)
            max_probs, targets_u = torch.max(probs_u, dim=-1)
            mask = max_probs.ge(self.conf_thre).int()

            for i, name in enumerate(domlabel_u):      
                domain_logit_avg[name].append(outputs_u[i].cpu())
                domain_feat_avg[name].append(img_feat[i].cpu())       
                self.main_domain[name] += mask[i] 
                keys = set(keys).union(set(domlabel_u))
                

        self.main_domain = sorted(self.main_domain.keys(), key=lambda item: self.main_domain[item], reverse=True)[0]
        
        for name in list(keys):
            self.domain_feat_avg[name] = torch.stack(domain_feat_avg[name], dim=0).mean(dim=0)  # new precision
            self.domain_feat_avg[name] = self.domain_feat_avg[name].unsqueeze(dim=0)
        self.domain_feat_avg['avg'] = torch.cat([torch.stack(domain_feat_avg[name]) for name in keys], dim=0).mean(dim=0)
        self.domain_feat_avg['avg'] = self.domain_feat_avg['avg'].unsqueeze(dim=0)
        print('calibrate main domain: ', self.main_domain)

    
        for name in list(keys):
            # image calibration
            self.domain_logit_avg[name] = self.calculate_large_tensor(domain_logit_avg[name])
            self.domain_logit_avg[name] = self.domain_logit_avg[name].unsqueeze(dim=0).cuda()
            self.domimg_bias_logits[name] = self.domain_logit_avg[name]

            # text calibration
            self.domain_feat_avg[name] = - self.domain_feat_avg[name] + self.domain_feat_avg['avg']  
            self.domain_feat_avg[name] = self.domain_feat_avg[name] / self.domain_feat_avg[name].norm(dim=-1, keepdim=True)    
            self.domtext_bias_features[name] = self.domain_feat_avg[name].cuda()


        if 'ensemble' in self.cfg.TRAINER.CALIBRATE_TEXT:
            self.text_bias_features = sum(self.domtext_bias_features.values()).cuda()
                    
        



    def calculate_large_tensor(self, large_list):
        interval = 1000
        interval_list = [sum(large_list[i:i+interval])/interval for i in range(0, len(large_list), interval)]
        weighted_list = sum([len(large_list[i:i+interval])/interval for i in range(0, len(large_list), interval)])
        return sum(interval_list) / weighted_list


    def parse_batch_train_dompred(self, batch_u):
        input_u = batch_u["img"]  
        index_u = batch_u['index']
        label_u = batch_u["label"]
        domlabel_u = batch_u["domlabel"]
        domname_list_u = batch_u["domain"]

        input_u = input_u.to(self.device)
        index_u = index_u.to(self.device)
        label_u = label_u.to(self.device)
        domlabel_u = domlabel_u.numpy()

        return input_u, index_u, label_u, domlabel_u, domname_list_u
    

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        import copy
        cur_evaluator = copy.deepcopy(self.evaluator)

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.test_loader is not None:
            data_loader = self.test_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        import time
        start_time = time.time()
        print(f"Evaluate on the *{split}* set")

        len_dom = 0
        if type(data_loader) is dict:
            accuracys = defaultdict(list)
            for domain, loader in data_loader.items():
                print(f'Test Accuracy on {domain}' )
                cur_evaluator.reset()
                for batch_idx, batch in enumerate(tqdm(loader)):
                    input, label = self.parse_batch_test(batch)
                    domlabel = batch['domlabel']   
                    # output = self.model_inference(input, domain)      #! domain name: ground truth
                    output = self.model_inference(input, domlabel)      #! domain pred: cluster
                    self.evaluator.process(output, label, len_dom)
                    cur_evaluator.process(output, label, len_dom)
                
                cur_results = cur_evaluator.evaluate(domain=domain)
                accuracys[domain] = float(format(cur_results['accuracy'], '.2f'))
                
            results = self.evaluator.evaluate()

            for k, v in results.items():
                tag = f"{split}/{k}"
                self.write_scalar(tag, v, self.epoch)

            average = sum(accuracys.values())/len(accuracys.values())
            print(
                "=> summary results \n"
                f"{accuracys} \n"
                f"average results: {average:.2f}%"
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time} seconds")
            return 0   

        else:     
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                input, label = self.parse_batch_test(batch)
                domlabel = batch['domlabel']    # upper bound 
                output = self.model_inference(input, domlabel)
                self.evaluator.process(output, label, len_dom)

            results = self.evaluator.evaluate()

            for k, v in results.items():
                tag = f"{split}/{k}"
                self.write_scalar(tag, v, self.epoch)

            return list(results.values())[0]

   

# prompt ensemble: IMAGENET_TEMPLATES_SELECT
@TRAINER_REGISTRY.register()
class UMFC_ensemble(UMFC):
    templates = IMAGENET_TEMPLATES_SELECT

    def build_model(self):
        cfg = self.cfg
        self.lab2cname = self.dm.dataset.lab2cname
        classnames = self.dm.dataset.classnames
        self.classnames = classnames
        self.domains = self.dm.dataset.domain_list
        self.remove_classes = []
        self.superclass = []
        self.eval_train_loader_u = self.dm.eval_train_loader_u

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)
        
        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates += ["a photo of a {}."]
        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")
        print('templates: ', self.templates)

        mean_text_features = 0
        with torch.no_grad():
            for i, temp in enumerate(self.templates):
                prompts = [temp.format(c.replace("_", " ")) for c in classnames]
                prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
                text_features = clip_model.encode_text(prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                mean_text_features = mean_text_features + text_features
            mean_text_features = mean_text_features / num_temp

        self.text_features = mean_text_features
        self.clip_model = clip_model

        self.dtype = clip_model.dtype
        self.T = 1.0
        self.conf_thre = 0.95
        self.alpha = cfg.TRAINER.CALIBRATE_IMG_WEIGHT
        self.beta = cfg.TRAINER.CALIBRATE_TEXT_WEIGHT

        import time
        start_time = time.time()  

        self.calibrate(cfg, classnames)
        
        end_time = time.time()  
        execution_time = end_time - start_time  
        print(f"Calibration time: {execution_time} seconds")
        

# search hyper-parameter: alpha, beta
@TRAINER_REGISTRY.register()
class UMFC_summary(UMFC):
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        import copy
        cur_evaluator = copy.deepcopy(self.evaluator)

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.test_loader is not None:
            data_loader = self.test_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        len_dom = 0
        if type(data_loader) is dict:
            accuracys = defaultdict(float)
            summary_accuracy = defaultdict(list)
            alpha_list = np.arange(0.1, 1.0, 0.2)
            beta_list = np.arange(0.1, 1.0, 0.2)

            for alpha in alpha_list:
                self.alpha = alpha
                for beta in beta_list:
                    self.beta = beta

                    for domain, loader in data_loader.items():
                        print(f'Test Accuracy on {domain}' )
                        cur_evaluator.reset()
                        for batch_idx, batch in enumerate(tqdm(loader)):
                            input, label = self.parse_batch_test(batch)
                            domlabel = batch['domlabel']    # domain pred: cluster
            
                            output = self.model_inference(input, domlabel)      
                            self.evaluator.process(output, label, len_dom)
                            cur_evaluator.process(output, label, len_dom)
                        
                        cur_results = cur_evaluator.evaluate(domain=domain)
                        accuracys[domain] = float(format(cur_results['accuracy'], '.2f'))
                        summary_accuracy[domain].append(accuracys[domain])
                        
                    results = self.evaluator.evaluate()

                    for k, v in results.items():
                        tag = f"{split}/{k}"
                        self.write_scalar(tag, v, self.epoch)

                    average = float(format(sum(accuracys.values())/len(accuracys.values()), '.2f'))
                    summary_accuracy['average'].append(average)
                    print(
                        f"=> summary results on alpha{alpha:.2f} - beta{beta:.2f}\n"
                        f"{accuracys} \n"
                        f"average results: {average:.2f}%"
                    )

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


# for Transductive Learning: UMFC + CLIP-E
@TRAINER_REGISTRY.register()
class UMFC_ensemble_summary(UMFC_ensemble):
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        import copy
        cur_evaluator = copy.deepcopy(self.evaluator)

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.test_loader is not None:
            data_loader = self.test_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        len_dom = 0
        if type(data_loader) is dict:
            accuracys = defaultdict(float)
            summary_accuracy = defaultdict(list)
            alpha_list = np.arange(0.1, 1.2, 0.2)
            beta_list = np.arange(0.1, 1.0, 0.2)

            for alpha in alpha_list:
                self.alpha = alpha
                for beta in beta_list:
                    self.beta = beta

                    for domain, loader in data_loader.items():
                        print(f'Test Accuracy on {domain}' )
                        cur_evaluator.reset()
                       
                        for batch_idx, batch in enumerate(tqdm(loader)):
                            input, label = self.parse_batch_test(batch)
                            domlabel = batch['domlabel']    #! domain pred: cluster name
            
                            # output = self.model_inference(input, domain)      #! domain name: ground truth
                            output = self.model_inference(input, domlabel)      #! domain pred: cluster
                            self.evaluator.process(output, label, len_dom)
                            cur_evaluator.process(output, label, len_dom)
                        
                        cur_results = cur_evaluator.evaluate(domain=domain)
                        accuracys[domain] = float(format(cur_results['accuracy'], '.2f'))
                        summary_accuracy[domain].append(accuracys[domain])
                        
                    results = self.evaluator.evaluate()

                    for k, v in results.items():
                        tag = f"{split}/{k}"
                        self.write_scalar(tag, v, self.epoch)

                    average = float(format(sum(accuracys.values())/len(accuracys.values()), '.2f'))
                    summary_accuracy['average'].append(average)
                    print(
                        f"=> summary results on alpha{alpha:.2f} - beta{beta:.2f}\n"
                        f"{accuracys} \n"
                        f"average results: {average:.2f}%"
                    )

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


# for unsupervised calibration: UMFC
@TRAINER_REGISTRY.register()
class UMFC_trainu_summary(UMFC_summary):

    @torch.no_grad()
    def get_image_bias(self):
        domain_feat_avg = defaultdict(list)
        domain_logit_avg = defaultdict(list)
        self.domain_img_avg = {}
        self.domain_feat_avg = {}
        self.domain_logit_avg = {}
        self.domain_prob_avg = {}
        self.main_domain = defaultdict(int)
        from tqdm import tqdm
        loader = self.eval_train_loader_u # change dataloader
        self.set_model_mode('eval')
        keys = []

        for batch_idx, batch in enumerate(tqdm(loader, desc="Collecting domain average information")):
            parsed_data = self.parse_batch_train_dompred(batch)
            input_u, index_u, label_u, domlabel_u, domname_list_u = parsed_data

            outputs_u, img_feat = self.get_logits_features(input_u)
            probs_u = torch.softmax(outputs_u, dim=-1)
            max_probs, targets_u = torch.max(probs_u, dim=-1)
            mask = max_probs.ge(self.conf_thre).int()

            for i, name in enumerate(domlabel_u):      
                domain_logit_avg[name].append(outputs_u[i].cpu())
                domain_feat_avg[name].append(img_feat[i].cpu())       
                self.main_domain[name] += mask[i] 
                keys = set(keys).union(set(domlabel_u))
                

        self.main_domain = sorted(self.main_domain.keys(), key=lambda item: self.main_domain[item], reverse=True)[0]
        
        for name in list(keys):
            self.domain_feat_avg[name] = torch.stack(domain_feat_avg[name], dim=0).mean(dim=0) 
            self.domain_feat_avg[name] = self.domain_feat_avg[name].unsqueeze(dim=0)
        self.domain_feat_avg['avg'] = torch.cat([torch.stack(domain_feat_avg[name]) for name in keys], dim=0).mean(dim=0)
        self.domain_feat_avg['avg'] = self.domain_feat_avg['avg'].unsqueeze(dim=0)
        print('calibrate main domain: ', self.main_domain)

    
        for name in list(keys):
            # image calibration
            self.domain_logit_avg[name] = self.calculate_large_tensor(domain_logit_avg[name])
            self.domain_logit_avg[name] = self.domain_logit_avg[name].unsqueeze(dim=0).cuda()
            self.domimg_bias_logits[name] = self.domain_logit_avg[name]

            # text calibration
            self.domain_feat_avg[name] = - self.domain_feat_avg[name] + self.domain_feat_avg['avg']  
            self.domain_feat_avg[name] = self.domain_feat_avg[name] / self.domain_feat_avg[name].norm(dim=-1, keepdim=True)    
            self.domtext_bias_features[name] = self.domain_feat_avg[name].cuda()


        if 'ensemble' in self.cfg.TRAINER.CALIBRATE_TEXT:
            self.text_bias_features = sum(self.domtext_bias_features.values()).cuda()
                    
        

# for unsupervised calibration: UMFC + CLIP-E
@TRAINER_REGISTRY.register()
class UMFC_ensemble_trainu_summary(UMFC_ensemble):
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        import copy
        cur_evaluator = copy.deepcopy(self.evaluator)

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.test_loader is not None:
            data_loader = self.test_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        len_dom = 0
        if type(data_loader) is dict:
            accuracys = defaultdict(float)
            summary_accuracy = defaultdict(list)
            alpha_list = np.arange(0.0, 1.2, 0.1)
            beta_list = np.arange(0.0, 1.0, 0.1)

            for alpha in alpha_list:
                self.alpha = alpha
                for beta in beta_list:
                    self.beta = beta

                    for domain, loader in data_loader.items():
                        print(f'Test Accuracy on {domain}' )
                        cur_evaluator.reset()

                        for batch_idx, batch in enumerate(tqdm(loader)):
                            input, label = self.parse_batch_test(batch)
                            domlabel = batch['domlabel']    #! domain pred: cluster name
                            
                            # output = self.model_inference(input, domain)      #! domain name: ground truth
                            output = self.model_inference(input, domlabel)      #! domain pred: cluster
                            self.evaluator.process(output, label, len_dom)
                            cur_evaluator.process(output, label, len_dom)
                        
                        cur_results = cur_evaluator.evaluate(domain=domain)
                        accuracys[domain] = float(format(cur_results['accuracy'], '.2f'))
                        summary_accuracy[domain].append(accuracys[domain])
                        
                    results = self.evaluator.evaluate()

                    for k, v in results.items():
                        tag = f"{split}/{k}"
                        self.write_scalar(tag, v, self.epoch)

                    average = float(format(sum(accuracys.values())/len(accuracys.values()), '.2f'))
                    summary_accuracy['average'].append(average)
                    print(
                        f"=> summary results on alpha{alpha:.2f} - beta{beta:.2f}\n"
                        f"{accuracys} \n"
                        f"average results: {average:.2f}%"
                    )

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
    
    @torch.no_grad()
    def get_image_bias(self):
        domain_feat_avg = defaultdict(list)
        domain_logit_avg = defaultdict(list)
        self.domain_img_avg = {}
        self.domain_feat_avg = {}
        self.domain_logit_avg = {}
        self.domain_prob_avg = {}
        self.main_domain = defaultdict(int)
        from tqdm import tqdm
        loader = self.eval_train_loader_u # change dataloader
        self.set_model_mode('eval')
        keys = []

        for batch_idx, batch in enumerate(tqdm(loader, desc="Collecting domain average information")):
            parsed_data = self.parse_batch_train_dompred(batch)
            input_u, index_u, label_u, domlabel_u, domname_list_u = parsed_data

            outputs_u, img_feat = self.get_logits_features(input_u)
            probs_u = torch.softmax(outputs_u, dim=-1)
            max_probs, targets_u = torch.max(probs_u, dim=-1)
            mask = max_probs.ge(self.conf_thre).int()

            for i, name in enumerate(domlabel_u):      
                domain_logit_avg[name].append(outputs_u[i].cpu())
                domain_feat_avg[name].append(img_feat[i].cpu())       
                self.main_domain[name] += mask[i] 
                keys = set(keys).union(set(domlabel_u))
                

        self.main_domain = sorted(self.main_domain.keys(), key=lambda item: self.main_domain[item], reverse=True)[0]
        
        for name in list(keys):
            self.domain_feat_avg[name] = torch.stack(domain_feat_avg[name], dim=0).mean(dim=0) 
            self.domain_feat_avg[name] = self.domain_feat_avg[name].unsqueeze(dim=0)
        self.domain_feat_avg['avg'] = torch.cat([torch.stack(domain_feat_avg[name]) for name in keys], dim=0).mean(dim=0)
        self.domain_feat_avg['avg'] = self.domain_feat_avg['avg'].unsqueeze(dim=0)
        print('calibrate main domain: ', self.main_domain)

    
        for name in list(keys):
            # image calibration
            self.domain_logit_avg[name] = self.calculate_large_tensor(domain_logit_avg[name])
            self.domain_logit_avg[name] = self.domain_logit_avg[name].unsqueeze(dim=0).cuda()
            self.domimg_bias_logits[name] = self.domain_logit_avg[name]

            # text calibration
            self.domain_feat_avg[name] = - self.domain_feat_avg[name] + self.domain_feat_avg['avg']  
            self.domain_feat_avg[name] = self.domain_feat_avg[name] / self.domain_feat_avg[name].norm(dim=-1, keepdim=True)    
            self.domtext_bias_features[name] = self.domain_feat_avg[name].cuda()


        if 'ensemble' in self.cfg.TRAINER.CALIBRATE_TEXT:
            self.text_bias_features = sum(self.domtext_bias_features.values()).cuda()
                    
        