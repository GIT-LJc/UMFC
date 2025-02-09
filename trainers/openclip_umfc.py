import torch
import torch.nn as nn
import torch.nn.functional as F

from dassl.engine import TRAINER_REGISTRY
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip

from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT, DOMAINNET_TEMPLATES

from tqdm import tqdm
from collections import defaultdict
import numpy as np  
import os
from dassl.utils import mkdir_if_missing

from .zsclip import ZeroshotCLIP
from .umfc_UC import UMFC, UMFC_ensemble
import open_clip

def load_openclip_to_cpu(cfg):
    arch = cfg.ARCH
    arch_dict = {'ViT-B-32':'../scaling-laws-openclip/Model-B-32_Data-2B_Samples-34B_lr-1e-3_bs-79k.pt', 'ViT-B-16':'../scaling-laws-openclip/Model-B-16_Data-2B_Samples-34B_lr-1e-3_bs-88k.pt', 'ViT-H-14':'../scaling-laws-openclip/Model-H-14_Data-2B_Samples-34B_lr-5e-4_bs-79k.pt', 'ViT-B-16-80M':'../scaling-laws-openclip/Model-B-16_Data-80M_Samples-13B_lr-1e-3_bs-88k.pt', 'ViT-B-16-400M':'../scaling-laws-openclip/Model-B-16_Data-400M_Samples-13B_lr-5e-4_bs-33k.pt'}
    path = arch_dict[arch]
    # arch='ViT-B-16'
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=path)
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer(arch)
    return model, tokenizer


@TRAINER_REGISTRY.register()
class OpenCLIP(ZeroshotCLIP):
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.classnames = classnames
        domains = self.dm.dataset.domains

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, tokenizer = load_openclip_to_cpu(cfg)
        clip_model.to(self.device)
        
        temp = "a photo of a {}."
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]

        print(f"Prompts: {prompts}")
        prompts = torch.cat([tokenizer(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model


@TRAINER_REGISTRY.register()
class OpenCLIP_calibrate_v3(UMFC):
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
        clip_model, tokenizer = load_openclip_to_cpu(cfg)
        clip_model.to(self.device)
        
        prompts = [self.temp.format(c.replace("_", " ")) for c in self.classnames]

        print(f"Prompts: {prompts}")
        prompts = torch.cat([tokenizer(p) for p in prompts])
        prompts = prompts.to(self.device)
        
        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)

        self.text_features = text_features
        self.clip_model = clip_model

        self.dtype = torch.float16
        self.T = 1.0
        self.conf_thre = 0.95
        self.alpha = cfg.TRAINER.CALIBRATE_IMG_WEIGHT
        self.beta = cfg.TRAINER.CALIBRATE_TEXT_WEIGHT
        self.calibrate(cfg, classnames)
        

@TRAINER_REGISTRY.register()
class OpenCLIP_calibrate_v3_summary(OpenCLIP_calibrate_v3):
    
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



@TRAINER_REGISTRY.register()
class OpenCLIP_calibrate_v3_ensemble(UMFC_ensemble):
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
        clip_model, tokenizer = load_openclip_to_cpu(cfg)
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
                prompts = torch.cat([tokenizer(p) for p in prompts]).to(self.device)
                text_features = clip_model.encode_text(prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                mean_text_features = mean_text_features + text_features
            mean_text_features = mean_text_features / num_temp
        # mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model

        # self.dtype = clip_model.dtype
        self.dtype = torch.float16
        self.T = 1.0
        self.conf_thre = 0.95
        self.alpha = cfg.TRAINER.CALIBRATE_IMG_WEIGHT
        self.beta = cfg.TRAINER.CALIBRATE_TEXT_WEIGHT
        self.calibrate(cfg, classnames)
        


@TRAINER_REGISTRY.register()
class OpenCLIP_calibrate_v3_ensemble_summary(OpenCLIP_calibrate_v3_ensemble):
    
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
            # alpha_list = np.arange(0.0, 1.1, 0.3)
            # beta_list = np.arange(0.0, 0.7, 0.2)
            alpha_list = np.arange(0.5, 0.8, 0.1)
            beta_list = np.arange(0.8, 1.1, 0.1)

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



@TRAINER_REGISTRY.register()
class OpenCLIP_calibrate_v3_ensemble_trainu_summary(OpenCLIP_calibrate_v3_ensemble):
        # calibrate名字中含‘img’
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
            # alpha_list = np.arange(0.0, 1.1, 0.3)
            # beta_list = np.arange(0.0, 0.7, 0.2)
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
        domain_img_avg = defaultdict(list)
        domain_feat_avg = defaultdict(list)
        domain_logit_avg = defaultdict(list)
        domain_prob_avg = defaultdict(list)
        self.domain_img_avg = {}
        self.domain_feat_avg = {}
        self.domain_logit_avg = {}
        self.domain_prob_avg = {}
        self.main_domain = defaultdict(int)
        from tqdm import tqdm
        loader = self.eval_train_loader_u   
        self.set_model_mode('eval')
    
        for batch_idx, batch in enumerate(tqdm(loader, desc="Collecting domain average information")):
            parsed_data = self.parse_batch_train_dompred(batch)
            input_u, index_u, label_u, domlabel_u, domname_list_u = parsed_data

            outputs_u, img_feat = self.get_logits_features(input_u)
            probs_u = torch.softmax(outputs_u, dim=-1)
            max_probs, targets_u = torch.max(probs_u, dim=-1)
            mask = max_probs.ge(self.conf_thre).int()
   
            for i, name in enumerate(domlabel_u):      #! domname_list_u is a list of domain pred (clustered domain)
                domain_logit_avg[name].append(outputs_u[i].cpu())
                domain_feat_avg[name].append(img_feat[i].cpu())       
                self.main_domain[name] += mask[i] 
                keys = set(keys).union(set(domlabel_u))
                

        # ! self.main_domain is used for feature calibration
        self.main_domain = sorted(self.main_domain.keys(), key=lambda item: self.main_domain[item], reverse=True)[0]
        
        print('calibrate img keys: ', keys)
       
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
                    
  


