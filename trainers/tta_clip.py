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

from .zsclip import load_clip_to_cpu, ZeroshotCLIP_calibrate_v3

@TRAINER_REGISTRY.register()
class ZeroshotCLIP_TTC(ZeroshotCLIP_calibrate_v3):
    '''test-time calibration with pre clustering'''

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
        
        self.calibrate_init(cfg, classnames) # collect the calibration information
      
    def calibrate_init(self, cfg, classnames):
        # initialize bias
        self.image_bias_logits = torch.zeros(len(classnames)).to('cuda')
        self.text_bias_logits = torch.zeros(len(classnames)).to('cuda')
        self.image_bias_features = 0
        self.text_bias_features = 0

        self.domimg_bias_logits = {}
        self.domimg_bias_features = {}
        self.domtext_bias_logits = {}
        self.domtext_bias_features = {}
        self.domimg_bias_features_v6 = {}  # for v6

        # 可以用来存储每个batch的domain_img_avg等，仍然以list形式（memory）
        self.domain_img = defaultdict(list)
        self.domain_feat = defaultdict(list)
        self.domain_logit = defaultdict(list)
        self.domain_prob = defaultdict(list)
        # 下面这个是上面的平均, 可以用来存储截止目前batch的平均domain_img_avg等，每个values都只是一个值（EMA update形式）
        self.domain_img_avg = defaultdict(float)
        self.domain_feat_avg = defaultdict(float)
        self.domain_logit_avg = defaultdict(float)
        self.domain_prob_avg = defaultdict(float)

        # self.main_domain = defaultdict(int)
        self.main_domain_list = defaultdict(list)
        self.dim = self.text_features.shape[1]

        for i in range(6):
            self.domimg_bias_logits[i] = torch.zeros(1, len(classnames), dtype=self.dtype).to('cuda')
            self.domimg_bias_features[i] = torch.zeros(1, self.dim, dtype=self.dtype).to('cuda')
            self.domtext_bias_logits[i] = torch.zeros(1, len(classnames), dtype=self.dtype).to('cuda')
            self.domtext_bias_features[i] = torch.zeros(1, self.dim, dtype=self.dtype).to('cuda')
            self.domimg_bias_features_v6[i] = torch.zeros(1, self.dim, dtype=self.dtype).to('cuda')

        with torch.no_grad():
            if cfg.TRAINER.CALIBRATE_IMG == 'image_zero':
                print('calibrating image')
                zero_image = torch.zeros(1, 3, 224, 224).cuda()
                image_bias_logits = self.forward(zero_image.type(self.dtype))
                self.image_bias_logits = torch.softmax(image_bias_logits, dim=-1)
                self.draw_bias(classnames)
            elif cfg.TRAINER.CALIBRATE_IMG == 'image_inf':
                print('calibrating inf image')
                inf_image = torch.ones(1, 3, 224, 224) * 300
                inf_image = inf_image.cuda()
                image_bias_logits = self.forward(inf_image.type(self.dtype))
                self.image_bias_logits = torch.softmax(image_bias_logits, dim=-1)
                self.draw_bias(classnames)

    
    def pred_domain(self, image):
        domlabel = self.pred_model(image)
        self.pred_model.update()    #! pred_model的更新方式
        return domlabel

    def parse_batch_train_dompred(self, batch_u):
        input_u = batch_u["img"]  # weak augmentation
        index_u = batch_u['index']
        label_u = batch_u["label"]
        domname_list_u = batch_u["domain"]

        input_u = input_u.to(self.device)
        index_u = index_u.to(self.device)
        label_u = label_u.to(self.device)
        domlabel_u = self.domlabel.numpy()

        return input_u, index_u, label_u, domlabel_u, domname_list_u
    
    # todo : create a class for auto-update


    # calibrate名字中含‘img’, batch update
    @torch.no_grad()
    def get_image_bias(self, batch):
        domain_img_batch = defaultdict(list)
        domain_feat_batch = defaultdict(list)
        domain_logit_batch = defaultdict(list)

        self.set_model_mode('eval')
        calibrate_type = self.cfg.TRAINER.CALIBRATE_IMG
        
        need_feature_type = ['img_featlogit', 'img_feature', 'img_feature_main', 'img_feature_main2', 'img_feature_norm', 'img_feature_main_norm', 'img_feature_main2_norm', 'img_featlogit_main', 'img_featlogit_main2', 'img_feature_calibrate', 'img_feature_calibrate_norm', 'img_feature_avg', 'img_feature_avg_norm']

        # for batch_idx, batch in enumerate(tqdm(loader, desc="Collecting domain average information")):
        parsed_data = self.parse_batch_train_dompred(batch)
        input_u, index_u, label_u, domlabel_u, domname_list_u = parsed_data

        outputs_u, img_feat = self.get_logits_features(input_u)
        probs_u = torch.softmax(outputs_u, dim=-1)
        max_probs, targets_u = torch.max(probs_u, dim=-1)
        mask = max_probs.ge(self.conf_thre).int()
        # for i, name in enumerate(domname_list_u):    #! domname_list_u is a list of domain names (ground truth)
        for i, name in enumerate(domlabel_u):      #! domlabel_u is a list of domain pred (clustered domain)
            # self.main_domain_list[name] += mask[i] 
            # import pdb; pdb.set_trace()
            self.main_domain_list[name].append(mask[i])

            if calibrate_type == 'img_image':
                domain_img_batch[name].append(input_u[i].cpu())
            elif calibrate_type in need_feature_type:
                domain_feat_batch[name].append(img_feat[i].cpu())
            elif calibrate_type == 'img_logit':
                domain_logit_batch[name].append(outputs_u[i].cpu())
            elif 'none' not in self.cfg.TRAINER.CALIBRATE_IMG:
                raise ValueError(f'Invalid calibration method {self.cfg.TRAINER.CALIBRATE_IMG}')
            
            if ('img2text' in self.cfg.TRAINER.CALIBRATE_TEXT) and (calibrate_type not in need_feature_type):
                domain_feat_batch[name].append(img_feat[i].cpu())
                
        # 截止到此，是完成了对当前一个batch的数据的统计
        return domain_img_batch, domain_feat_batch, domain_logit_batch

    # update bias each batch
    def update_img_bias(self, batch_information):   
        self.main_domain = sorted(self.main_domain_list.keys(), key=lambda item: sum(self.main_domain_list[item])/(len(self.main_domain_list[item]) + 1e-5), reverse=True)[0]     # self.main_domain is used for feature calibration/alignment，在batch到来的场景下main_domain的确定方式未必合理，可以用比例来修正

        domain_img_batch, domain_feat_batch, domain_logit_batch =  batch_information   # 导入一个batch的统计信息: current information
        update_keys = self.get_keys(batch_information)
        # import pdb; pdb.set_trace()
        
        # ttc_update type
        if self.cfg.TTC_UPDATE == 'memory':
            if self.cfg.TRAINER.CALIBRATE_IMG == 'img_image':
                self.domain_img_avg = self.dict_add_list(self.domain_img, domain_img_batch, self.domain_img_avg)
            elif 'feat' in self.cfg.TRAINER.CALIBRATE_IMG:
                self.domain_feat_avg = self.dict_add_list(self.domain_feat, domain_feat_batch, self.domain_feat_avg)
            elif self.cfg.TRAINER.CALIBRATE_IMG == 'img_logit':
                self.domain_logit_avg = self.dict_add_list(self.domain_logit, domain_logit_batch, self.domain_logit_avg)
            if 'img2text' in self.cfg.TRAINER.CALIBRATE_TEXT:
                self.domain_feat_avg = self.dict_add_list(self.domain_feat, domain_feat_batch, self.domain_feat_avg)
        elif self.cfg.TTC_UPDATE == 'ema':
            if self.cfg.TRAINER.CALIBRATE_IMG == 'img_image':
                self.domain_img_avg = self.dict_update_list(self.domain_img_avg, domain_img_batch)
            elif 'feat' in self.cfg.TRAINER.CALIBRATE_IMG:
                self.domain_feat_avg = self.dict_update_list(self.domain_feat_avg, domain_feat_batch)
            elif self.cfg.TRAINER.CALIBRATE_IMG == 'img_logit':
                self.domain_logit_avg = self.dict_update_list(self.domain_logit_avg, domain_logit_batch)
            if 'img2text' in self.cfg.TRAINER.CALIBRATE_TEXT:
                self.domain_feat_avg = self.dict_update_list(self.domain_feat_avg, domain_feat_batch)
        # import pdb; pdb.set_trace()

        # for name in list(self.keys):
        for name in update_keys:
            if self.cfg.TRAINER.CALIBRATE_IMG == 'img_image':
                # self.domain_img_avg[name] = torch.stack(domain_img_avg[name], dim=0).mean(dim=0)  # new precision
                # self.domain_img_avg[name] = self.domain_img_avg[name].unsqueeze(dim=0)
                logits = self.forward(self.domain_img_avg[name].cuda())
                self.domimg_bias_logits[name] = logits

            elif 'img_featlogit' in self.cfg.TRAINER.CALIBRATE_IMG:
                # 从平均Feature产生的logits角度角度进行校准 - calculate the cosine similarity between the domain feature and the text feature
                # self.domain_feat_avg[name] = self.domain_feat_avg[name] / self.domain_feat_avg[name].norm(dim=-1, keepdim=True)
                domain_feat_avg = self.domain_feat_avg[name] / self.domain_feat_avg[name].norm(dim=-1, keepdim=True)   # 不要影响self.domain_feat_avg的信息
                text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
                logits = self.clip_model.logit_scale.exp() * domain_feat_avg.cuda() @ text_features.t()
                self.domimg_bias_logits[name] = logits

            elif self.cfg.TRAINER.CALIBRATE_IMG == 'img_logit':
                # self.domain_logit_avg[name] = self.calculate_large_tensor(domain_logit_avg[name])
                # self.domain_logit_avg[name] = self.domain_logit_avg[name].unsqueeze(dim=0).cuda()
                self.domimg_bias_logits[name] = self.domain_logit_avg[name].cuda()

                # if torch.isnan(self.domimg_bias_logits[name]).any() or torch.isinf(self.domimg_bias_logits[name]).any():
                #     print('domimg_bias_logits', torch.isnan(self.domimg_bias_logits[name]).any(),  torch.isinf(self.domimg_bias_logits[name]).any())
                #     import pdb; pdb.set_trace()


            elif 'img_feature' in self.cfg.TRAINER.CALIBRATE_IMG:
                # 直接从Feature角度进行校准，不考虑logits， 包括img_feature (not norm), img_feature_norm
                self.domimg_bias_features[name] = self.domain_feat_avg[name].cuda()

                if 'img_feature_main' in self.cfg.TRAINER.CALIBRATE_IMG:  #! align with main domain
                    print(f'align with main domain {self.main_domain}')
                    if 'img_feature_main2' in self.cfg.TRAINER.CALIBRATE_IMG:
                        if name != self.main_domain:
                            self.domimg_bias_features[name] = (self.domain_feat_avg[name] - self.domain_feat_avg[self.main_domain]).cuda()
                    else:
                        self.domimg_bias_features[name] = (self.domain_feat_avg[name] - self.domain_feat_avg[self.main_domain]).cuda()

                elif 'img_feature_avg' in self.cfg.TRAINER.CALIBRATE_IMG:  #! align with avg domain (NIPS23)
                    self.domimg_bias_features[name] = self.domain_feat_avg['avg'].cuda()
            # TODO： 上面是关于domimg_bias_features的确定方式

            # import pdb; pdb.set_trace()

            if 'img2text' in self.cfg.TRAINER.CALIBRATE_TEXT:   
                # 将其他domain的Feature与main domain的Feature作差，作为Feature shift，然后校准text_features
                
                if 'v2' in self.cfg.TRAINER.CALIBRATE_TEXT:   # norm之后进行特征相减
                    domain_feat_avg = self.domain_feat_avg[name].norm(dim=-1, keepdim=True) - self.domain_feat_avg[self.main_domain].norm(dim=-1, keepdim=True)
                # elif 'v3' in self.cfg.TRAINER.CALIBRATE_TEXT:
                elif 'v32' in self.cfg.TRAINER.CALIBRATE_TEXT:
                    # 'v32' self.domain_feat_avg['avg']用全部图像，减去全部图像的平均值，而不是减去main domain的平均值
                    domain_feat_avg = self.domain_feat_avg[name] - self.domain_feat_avg['avg']
                elif 'pig3' in self.cfg.TRAINER.CALIBRATE_TEXT:
                    domain_feat_avg = - self.domain_feat_avg[name] + self.domain_feat_avg['avg']
                elif 'v4' in self.cfg.TRAINER.CALIBRATE_TEXT:
                    domain_feat_avg = self.domain_feat_avg[name].norm(dim=-1, keepdim=True) - self.domain_feat_avg['avg'].norm(dim=-1, keepdim=True)

                elif 'v6' in self.cfg.TRAINER.CALIBRATE_TEXT:  # after normalize
                    self.domimg_bias_features_v6[name] = self.domain_feat_avg[name].cuda()
                    if 'main' in self.cfg.TRAINER.CALIBRATE_TEXT:
                        domain_feat_avg = self.text_features / self.text_features.norm(dim=-1, keepdim=True) - (self.domain_feat_avg['avg'] / self.domain_feat_avg[self.main_domain].norm(dim=-1, keepdim=True)).cuda()
                    elif 'avg' in self.cfg.TRAINER.CALIBRATE_TEXT:
                        domain_feat_avg = self.text_features / self.text_features.norm(dim=-1, keepdim=True) - (self.domain_feat_avg['avg'] / self.domain_feat_avg['avg'].norm(dim=-1, keepdim=True)).cuda()
                        
                else:
                    domain_feat_avg = self.domain_feat_avg[name] - self.domain_feat_avg[self.main_domain]

                self.domtext_bias_features[name] = domain_feat_avg.cuda()   # ps. domain_feat_avg替换原来的self.domain_feat_avg[name]是为了防止修改self.domain_feat_avg[name]的统计信息
                # import pdb; pdb.set_trace()
                if 'ensemble' in self.cfg.TRAINER.CALIBRATE_TEXT:
                    self.text_bias_features = sum(self.domtext_bias_features.values()).cuda()


    def update_text_bias(self):
        pass

    def get_keys(self, dicts):
        keys = []
        for d in dicts:
            keys = set(keys).union(set(d.keys()))
        return list(keys)


    def dict_add_list(self, dica, dicb, dicc):
        for k in self.keys:
            if k in dicb.keys():
                dica[k].extend(dicb[k])
                # dicc[k] = torch.stack(dica[k], dim=0).mean(dim=0).unsqueeze(dim=0)
                dicc[k] = self.calculate_large_tensor(dica[k]).unsqueeze(dim=0)
        # import pdb; pdb.set_trace()
        tmp = torch.cat([torch.stack(dica[name]) for name in dica.keys()], dim=0)
        dicc['avg'] = self.calculate_large_tensor(tmp).unsqueeze(dim=0)
        # if torch.isnan(self.domain_logit_avg['avg']).any() or torch.isinf(self.domain_logit_avg['avg']).any():
        #     print(torch.isnan(self.domain_logit_avg['avg']).any(), torch.isinf(self.domain_logit_avg['avg']).any())
        #     tmp = [torch.stack(dica[name]) for name in dica.keys()]
        #     avg = torch.cat(tmp, dim=0).mean(dim=0, keepdim=True)
        #     avg2 = torch.mean(torch.cat(tmp, dim=0), dim=0, keepdim=True)
        #     import pdb; pdb.set_trace()
        return dicc
    

    def dict_update_list(self, dica, dicb, batchid=4):
        # gamma = batchid / (batchid + 1)
        for k in dicb.keys():
            avg_b = torch.stack(dicb[k], dim=0).mean(dim=0).unsqueeze(dim=0)
            dica[k] = self.gamma * dica[k] + (1 - self.gamma) * avg_b
        dica['avg'] = torch.cat([dica[name] for name in dica.keys()], dim=0).mean(dim=0, keepdim=True)
        return dica

    # TODO：实际上是新的test 函数，要包括几个部分：得到domlabel，更新cluster model，得到特征，更新特征，做出矫正
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
        #! test-time 场景下按照domain到来的数据不是很合理了，主要是对domain label的确定会有较大影响，这一部分就暂且先不修改了
        if type(data_loader) is dict:
            accuracys = defaultdict(list)
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
            return 0   # 仅涉及测试阶段，无需保存current result

        else:   
            print(f'Test-Time Calibration Accuracy' )  
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                input, label = self.parse_batch_test(batch)
                domain = batch['domain']    # gt label, for test-acc 
                self.domain = domain
                # domlabel = self.pred_model(input)    # predicted domain label
                domlabel = batch["domlabel"]         # pre-cluster domain label
                self.domlabel = domlabel      # todo: 得到不同的domain label，只需要在这里修改即可

                batch_information = self.get_image_bias(batch)    # collect image bias information
                self.get_text_bias()           # collect image bias information

                # import pdb; pdb.set_trace()
                self.update_img_bias(batch_information)    # update image bias information
                # import pdb; pdb.set_trace()
                self.update_text_bias()    # update text bias information

                output = self.model_inference(input, domlabel)
                #output = self.model_inference(input)
                
                # output = self.model_inference(input, domain)
                # print(self.domain)
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



# clustering with kmeans
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
class ZeroshotCLIP_TTC_prototype(ZeroshotCLIP_TTC):
    '''test-time calibration with prototype clustering'''
    def __init__(self, cfg):
        super().__init__(cfg)


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
        # todo: 1. kmeans cluster

        import time
        start_time = time.time()
        start_epoch = self.cfg.TRAINER.UNLABELED_CLUSTERS
        start_features = None
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            domain = batch['domain']    # gt label, for test-acc 
            self.domain = domain

            # 对image feature聚类
            img_feature = self.get_image_features(input)

            # if batch_idx == 0:      # only for the first batch
            if batch_idx < start_epoch:      # only for the start batch  
                self.pred_model = prototype_cluster(n_clusters=self.cfg.TRAINER.UNLABELED_CLUSTERS)    # initialize the prototype cluster model
                # self.pred_model = prototype_cluster_torch(n_clusters=self.cfg.TRAINER.UNLABELED_CLUSTERS, device=self.device)    # initialize the prototype cluster model
                if batch_idx == 0:
                    start_features = img_feature
                else:
                    start_features = torch.cat((start_features, img_feature), dim=0)
                domlabel = self.pred_model.start(start_features)
                # domlabel = self.pred_model.start(img_feature)
            else:
                domlabel = self.pred_model.pred_update(img_feature)    # predicted domain label
            # domlabel = batch["domlabel"]         # pre-cluster domain label
            self.domlabel = domlabel      #! 得到不同的domain label，只需要在这里修改即可

            batch_information = self.get_image_bias(batch)    # collect image bias information
            self.get_text_bias()           # collect image bias information

            self.update_img_bias(batch_information)    # update image bias information
            self.update_text_bias()    # update text bias information

            output = self.model_inference(input, domlabel)
            
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

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        # import pdb; pdb.set_trace()
        

        return 0




@TRAINER_REGISTRY.register()
class ZeroshotCLIP_TTC_prototype_Summary(ZeroshotCLIP_TTC_prototype):
 

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
        alpha_list = [0.5, 0.6, 0.7]
        alpha_list = [0.7]
        beta_list = np.arange(0.1, 0.6, 0.2)
        if 'v6' in self.cfg.TRAINER.CALIBRATE_TEXT:
            # alpha_list = np.arange(0.4, 0.8, 0.1)
            # beta_list = np.arange(0.6, 1.0, 0.1)
            alpha_list = np.arange(0.2, 0.8, 0.1)
            beta_list = np.arange(0.4, 0.6, 0.1)
        for alpha in alpha_list:
            self.alpha = alpha
            for beta in beta_list:
                self.beta = beta
                #! initialize the calibrate bias
                self.calibrate_init(self.cfg, self.classnames)
                
                start_epoch = int(self.cfg.TRAINER.UNLABELED_CLUSTERS / self.cfg.DATALOADER.TEST.BATCH_SIZE)
                start_features = None

                # ? collect information from the first start_epoch, (when the number of samples is smaller than the cluster number M)
                for batch_idx, batch in enumerate(tqdm(data_loader)):
                    input, label = self.parse_batch_test(batch)
                    domain = batch['domain']    # gt label, for test-acc 
                    self.domain = domain

                    # 对image feature聚类
                    img_feature = self.get_image_features(input)

                    if batch_idx <= start_epoch:      # only for the start batch  
                        self.pred_model = prototype_cluster(n_clusters=self.cfg.TRAINER.UNLABELED_CLUSTERS)    # initialize the prototype cluster model
                        if batch_idx == 0:
                            start_features = img_feature
                        else:
                            start_features = torch.cat((start_features, img_feature), dim=0)
                        if batch_idx == start_epoch:
                            domlabel = self.pred_model.start(start_features)
                        else:
                            domlabel=[torch.tensor(0)]
                    else:
                        break

                    

                for batch_idx, batch in enumerate(tqdm(data_loader)):
                    input, label = self.parse_batch_test(batch)
                    domain = batch['domain']    # gt label, for test-acc 
                    self.domain = domain

                    # 对image feature聚类
                    img_feature = self.get_image_features(input)

                    # if batch_idx <= start_epoch:      # only for the start batch  
                    #     self.pred_model = prototype_cluster(n_clusters=self.cfg.TRAINER.UNLABELED_CLUSTERS)    # initialize the prototype cluster model
                    #     if batch_idx == 0:
                    #         start_features = img_feature
                    #     else:
                    #         start_features = torch.cat((start_features, img_feature), dim=0)
                    #     if batch_idx == start_epoch:
                    #         domlabel = self.pred_model.start(start_features)
                    #     else:
                    #         domlabel=[torch.tensor(0)]
                    #     # domlabel = self.pred_model.start(img_feature)
                    
                    # ? 原版
                    # if batch_idx == 0:      # only for the first batch
                    #     self.pred_model = prototype_cluster(n_clusters=self.cfg.TRAINER.UNLABELED_CLUSTERS)    # initialize the prototype cluster model
                    #     # self.pred_model = prototype_cluster_torch(n_clusters=self.cfg.TRAINER.UNLABELED_CLUSTERS, device=self.device)    # initialize the prototype cluster model
                    #     domlabel = self.pred_model.start(img_feature)
                    # else:
                        # domlabel = self.pred_model.pred_update(img_feature)    # predicted domain label
                    
                    domlabel = self.pred_model.pred_update(img_feature)    # predicted domain label
                    self.domlabel = domlabel

                    # import pdb; pdb.set_trace()
                    # if batch_idx >= start_epoch:
                    batch_information = self.get_image_bias(batch)    # collect image bias information
                    self.get_text_bias()           # collect image bias information

                    self.update_img_bias(batch_information)    # update image bias information
                    self.update_text_bias()    # update text bias information

                    output = self.model_inference(input, domlabel)
                    # else:
                    #     output = self.single_inference(input)

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

        return 0   # 仅涉及测试阶段，无需保存current result

    def summary_domain(self, dica, dicb):
        for k, v in dicb.items():
            dica[k].append(v)
        return dica

