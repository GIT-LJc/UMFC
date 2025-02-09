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


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    print('model path:', model_path)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model



@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.eval_train_loader_u = self.dm.eval_train_loader_u
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.classnames = classnames
        domains = self.dm.dataset.domains

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)
        # import pdb; pdb.set_trace()
        
        temp = "a photo of a {}."
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]

        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits

    def get_features(self, image):
        image_features = self.clip_model.encode_image(image)        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        return image_features, text_features
    
    @torch.no_grad()
    def get_features_test(self, dataloader='trainu'):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        vt_fea = defaultdict(list)
        if dataloader == 'trainu':
            data_loader = self.eval_train_loader_u
        elif dataloader == 'val':
            data_loader = self.val_loader
        
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        for _, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            dom_label = batch["domlabel"]
            dom_label = dom_label.to(self.device)
            domain = batch["domain"]
            impath = batch["impath"]
            # import pdb; pdb.set_trace()
            # classname = batch["classname"]
            # get features
            if hasattr(self, 'model'):
                v_fea, t_fea = self.model.get_features(input)
            elif hasattr(self, 'get_features'):
                v_fea, t_fea = self.get_features(input)
            vt_fea['v_fea'].extend(v_fea)
            vt_fea['label'].extend(label)
            vt_fea['dom_label'].extend(dom_label)
            vt_fea['domain'].extend(domain)
            vt_fea['impath'].extend(impath)
            # vt_fea['classname'].extend(classname)
        
        vt_fea['t_fea'].extend(self.text_features)
        vt_fea['classnames'] = self.dm.dataset.classnames

        return vt_fea


    # ! add a method to draw features
    @torch.no_grad()
    def test_information(self, path='pred_statics.txt'):
        cls_nums = len(self.classnames)
        
        data_loader = self.test_loader

        if type(data_loader) is dict:
            accuracys = defaultdict(list)
            domain_counter = defaultdict(int)        # 伪标签样本的domain分布
            domain_counter_correct = defaultdict(int)    # 伪标签样本的正确的domain分布
            domain_acc = defaultdict(float)    # 伪标签样本的正确的domain分布
            for domain_name, loader in data_loader.items():
                print(f'Test Accuracy on {domain_name}' )
                count = 0
                all_count = 0
                pselab_gt = dict.fromkeys(range(cls_nums), 0)     # 伪标签样本的真实类别分布
                pselab_pred = dict.fromkeys(range(cls_nums), 0)    # 伪标签样本的预测类别分布
                pselab_correct = dict.fromkeys(range(cls_nums), 0)    # 伪标签样本的预测正确的个数
                pselab_acc = dict.fromkeys(range(cls_nums), 0.)    # 伪标签样本的预测正确的个数
                class_counter = dict.fromkeys(range(cls_nums), 0)       # 原始全部无标签样本的类别分布
                for batch_idx, batch in enumerate(loader):
                    input = batch["img"].to(self.device)
                    domain = batch['domain']
                    label_u = batch["label"]
                    output_u_w = self.model_inference(input)
                    pseudo_label = torch.softmax(output_u_w.detach() / 1.0, dim=-1)
                    max_probs, targets_u  = torch.max(pseudo_label, dim=-1)
                    # mask = max_probs.ge(self.threshold).float()

                    for d, l, t in zip(domain, label_u, targets_u):
                        class_counter[l.item()] += 1
                        
                        domain_counter[d] += 1 
                        pselab_gt[l.item()] += 1
                        pselab_pred[t.item()] += 1
                        if l == t.item():
                            domain_counter_correct[d] += 1
                            pselab_correct[l.item()] += 1
        
                for c in pselab_acc.keys():
                    if pselab_pred[c]:
                        pselab_acc[c] = (pselab_correct[c] / pselab_pred[c]) * 100 

                domain_acc[domain_name] = (domain_counter_correct[domain_name] / domain_counter[domain_name]) * 100

                output = {
                    "domain_counter": domain_counter,
                    "domain_counter_correct": domain_counter_correct,
                    "pselab_gt": pselab_gt,
                    "pselab_pred": pselab_pred,
                    "pselab_correct": pselab_correct,
                    "class_counter": class_counter,
                    "pselab_acc": pselab_acc,
                    "domain_acc": domain_acc
                }
                print('draw the statistical information')
                self.write_results(output, domain_name, path)


    def write_results(self, statics, domain, path='pred_statics.txt'):
        import os
        save_path = os.path.join(self.cfg.OUTPUT_DIR, path)
        # classname = statics['classname']
        pselab_gt = statics['pselab_gt']
        pselab_pred = statics['pselab_pred']
        pselab_correct = statics['pselab_correct']
        pselab_acc = statics['pselab_acc']
        domain_counter = statics['domain_counter']
        domain_counter_correct = statics['domain_counter_correct']
        domain_acc = statics['domain_acc'][domain]

        cls_psec, pselab_pred, cls_psec_str = self.sort_dict(pselab_pred)
        pselab_gt = [pselab_gt[d] for d in cls_psec]
        pselab_correct = [pselab_correct[d] for d in cls_psec]
        pselab_acc = [pselab_acc[d] for d in cls_psec]
        classname = [self.lab2cname[l2c] for l2c in cls_psec]

        tplt = "{:<30}\t{:<10}\t{:<10}\t{:<10}\t{:.2f}%"

        with open(save_path, "a+") as f:
            if hasattr(self, 'superclass'):
                f.write('super_classes: '+ str(self.superclass) + '\n')
                f.write( 'remove_casses: ' + str(self.remove_classes) + '\n')
            f.write(f"Domain: {domain}, Accuracy: {domain_acc:.2f}%\n")
            f.write('Category                 ground-truth       pred     correct    acc\n')
            
            f.write('\n')
            for name, t, p, c, a in zip(classname, pselab_gt, pselab_pred, pselab_correct, pselab_acc):
                f.write(tplt.format(name, t, p, c, a, chr(255)))
                f.write('\n')
            f.write('\n\n')
            
        print(f"Results are written to {save_path}")

    @torch.no_grad()
    def test(self):
        super().test('test')
        # self.test_information(path='pred_statics.txt')
 
    def sort_dict(self, dicdata):    
        # sort dict with values from max to min 
        by_value = sorted(dicdata.items(),key = lambda item:item[1],reverse=True)
        x = []
        y = []
        strx = []   # 用于画图
        for d in by_value:
            x.append(d[0])
            strx.append(str(d[0]))
            y.append(d[1])
        return x, y, strx

    def draw(self, statics):
        print('draw the statistical information...')
        from matplotlib import pyplot as plt
        import os
        label2cname = self.dm.dataset.lab2cname

        pselab_pred = statics['pselab_pred']
        pselab_correct = statics['pselab_correct']
        pselab_gt = statics['pselab_gt']
        domain_counter = statics['domain_counter']
        domain_counter_correct = statics['domain_counter_correct']
        class_counter = statics['class_counter']
        pselab_acc = statics['pselab_acc']

        plt.figure(figsize=(300,28), dpi=500)
        fig, ax = plt.subplots(4,1)
        fig.tight_layout()
        fig.suptitle(f'Statistical information on pseudo-labels', fontsize=9)

        ax[0].set_title(f'Domain information', fontsize=7)
        dom, dom_nums, _ = self.sort_dict(domain_counter)
        dom_nums_corr = [domain_counter_correct[d] for d in dom]
        ax[0].bar(dom, dom_nums, label='domain_counter')
        ax[0].bar(dom, dom_nums_corr, label='domain_counter_correct')
        ax[0].tick_params(axis='x', labelsize=5) 
        ax[0].set_xticklabels(ax[0].get_xticklabels(),rotation=90)
        ax[0].set_xlabel('domains')    #设置x轴标题
        ax[0].set_ylabel('number')   #设置Y1轴标题
        ax[0].legend(fontsize=5)


        cls_psec, psec_nums, cls_psec_str = self.sort_dict(pselab_correct)
        psegt_nums = [pselab_gt[d] for d in cls_psec]
        cls_nums = [class_counter[d] for d in cls_psec]
        x_label = [label2cname[l2c] for l2c in cls_psec]

        ax[1].set_title(f'Information on unlabeled data used', fontsize=7)
        # ax[1].bar(cls_psec_str, cls_nums, label=f'original true class:{np.mean(cls_nums):.2f}', width=1)
        ax[1].bar(cls_psec_str, psegt_nums, label=f'pseudo-label true class:{np.mean(psegt_nums):.2f}', width=1)
        ax[1].set_xticks([])
        # ax[1].bar(x_label, cls_nums, label=f'original true class:{np.mean(cls_nums):.2f}', width=1)
        # ax[1].bar(x_label, psegt_nums, label=f'pseudo-label true class:{np.mean(psegt_nums):.2f}', width=0.9)
        # ax[1].tick_params(axis='x', labelsize=4) 
        # ax[1].set_xticklabels(ax[1].get_xticklabels(),rotation=45)
        ax[1].set_xlabel('class names')    #设置x轴标题
        ax[1].set_ylabel('numbers')   #设置Y2轴标题


        pse_nums = [pselab_pred[d] for d in cls_psec]
        pse_accs = [pselab_acc[d] for d in cls_psec]
        ax[2].set_title(f'pseudo-label information', fontsize=7)
        ax[2].bar(cls_psec_str, pse_nums, label=f'pselab_pred:{np.mean(pse_nums):.2f}', width=1)
        ax[2].bar(cls_psec_str, psec_nums, label=f'pselab_pred_correct:{np.mean(psec_nums):.2f}', width=1)
        ax[2].set_xticks([])
        # ax[2].bar(x_label, pse_nums, label=f'pselab_pred:{np.mean(pse_nums):.2f}', width=0.9)
        # ax[2].bar(x_label, psec_nums, label=f'pselab_pred_correct:{np.mean(psec_nums):.2f}', width=0.9)
        # ax[2].tick_params(axis='x', labelsize=4) 
        # ax[2].set_xticklabels(ax[2].get_xticklabels(),rotation=45)
        ax[2].set_xlabel('class names')    #设置x轴标题
        ax[2].set_ylabel('numbers')   #设置Y2轴标题
        ax[2].legend(fontsize=6)

        ax[3].set_title(f'pseudo-label accuracy', fontsize=7)
        ax[3].scatter(pse_nums, pse_accs, s=1)
        ax[3].set_xlabel('numbers')    #设置x轴标题
        ax[3].set_ylabel('accuracy')   #设置Y2轴标题

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
        plt.savefig(os.path.join(self.cfg.OUTPUT_DIR, 'statistics_{}.png'.format(self.threshold)))
        plt.close()


        # 设置格式tplt，20代表间隔距离，可根据自己需要调整
        tplt = "{:<30}\t{:<10}\t{:<10}\t{:<10}\t{:.2f}%"
        # 按tplt格式写入抬头行
        with open(os.path.join(self.cfg.OUTPUT_DIR, 'statistics_{}.txt'.format(self.threshold)), 'w') as output_fp:
            # output_fp.write(tplt.format('Category', '真实标签分布', '预测分布', '预测正确分布', '伪标签准确率', chr(255)))
            output_fp.write('Category    真实标签分布   预测分布   预测正确分布   伪标签准确率')
            # 换行
            output_fp.write('\n')

            for name, psegt_num, pse_num, psec_num, pse_acc in zip(x_label, psegt_nums, pse_nums, psec_nums, pse_accs):
                output_fp.write(tplt.format(name, psegt_num, pse_num, psec_num, pse_acc, chr(255)))
                output_fp.write('\n')

            output_fp.write('\n 100 tail classes:\n')
            for name in x_label[-100:]:
                output_fp.write(name)
                output_fp.write(', ')
            output_fp.write('\n')
            
            output_fp.write('\n 50 tail classes:\n')
            for name in x_label[-50:]:
                output_fp.write(name)
                output_fp.write(', ')
            output_fp.write('\n')
        
        output_fp.close()




@TRAINER_REGISTRY.register()
class ZeroshotCLIP_calibrate_v3(ZeroshotCLIP):
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
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model


        # ground truth emsembel text embedding
        num_temp = len(self.templates)
        print(f"ground-truth Prompt ensembling (n={num_temp})")
        print('templates: ', self.templates)

        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features.detach()
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)
        self.gt_text_features = mean_text_features

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

    #"""
    def model_inference(self, image, domain=None):
        if domain is not None:
            dom_list = [d.item() for d in domain]

            self.image_bias_logits = torch.stack([self.domimg_bias_logits[d] for d in dom_list]).squeeze()
            self.text_bias_logits = torch.stack([self.domtext_bias_logits[d] for d in dom_list]).squeeze()
            self.image_bias_features = torch.stack([self.domimg_bias_features[d] for d in dom_list]).squeeze(dim=1)
            if 'ensemble' not in self.cfg.TRAINER.CALIBRATE_TEXT:    
                self.text_bias_features = torch.stack([self.domtext_bias_features[d] for d in dom_list]).squeeze(dim=1)
            
            if 'v6' in self.cfg.TRAINER.CALIBRATE_TEXT:
                if 'ensemimg' in self.cfg.TRAINER.CALIBRATE_TEXT:   #! ensemble image features 是没有意义的，相当于再重新加上bias中减去的平均
                    # self.img_avg_features = sum(self.domimg_bias_features_v6.values())/len(self.domimg_bias_features_v6)  # or torch.mean
                    self.img_avg_features = torch.mean(torch.stack(list(self.domimg_bias_features_v6.values())),dim=0)
                else:
                    self.img_avg_features = torch.stack([self.domimg_bias_features_v6[d] for d in dom_list]).squeeze(dim=1)
                
            # if self.text_bais_features.dim() == 1:
            #     self.text_bias_features = self.text_bias_features.unsqueeze(0)
        logits = self.calibrate_logits(image)
        return logits
    #"""

    """
    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits
    """

    def calibrate_logits(self, image):
        image_features = self.clip_model.encode_image(image)
        text_features = self.text_features

        if 'norm' in self.cfg.TRAINER.CALIBRATE_IMG:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            norm_bias = self.image_bias_features / self.image_bias_features.norm(dim=-1, keepdim=True)
            if torch.sum(torch.isnan(norm_bias).int()) != 0:
                mask = torch.sum(self.image_bias_features, dim=-1)
                bias_list = []
                for m, i, ni in zip(mask, self.image_bias_features, norm_bias):
                    if m == 0:
                        bias_list.append(torch.zeros_like(i))
                    else:
                        bias_list.append(ni)
                self.image_bias_features = torch.stack(bias_list)
                # import pdb; pdb.set_trace()
            else:
                self.image_bias_features = norm_bias

        # calibrate features
        ca_img_features = image_features - self.alpha * self.image_bias_features
        if 'img_feature_calibrate' not in self.cfg.TRAINER.CALIBRATE_IMG:
            ca_img_features = ca_img_features / ca_img_features.norm(dim=-1, keepdim=True)

        # ca_text_features = self.text_features - self.beta * self.text_bias_features
        # ca_text_features = ca_text_features / ca_text_features.norm(dim=-1, keepdim=True)
        # import pdb; pdb.set_trace()
        # ca_logits = torch.stack([self.clip_model.logit_scale.exp() * ca_img_features @ ctf.t() for ctf in ca_text_features], dim=0)

        # if 'feature' in self.cfg.TRAINER.CALIBRATE_TEXT :   
        if 'img2text_shift' in self.cfg.TRAINER.CALIBRATE_TEXT:
            if 'norm' in self.cfg.TRAINER.CALIBRATE_TEXT:
                text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
                norm_bias = self.text_bias_features / self.text_bias_features.norm(dim=-1, keepdim=True)
                if torch.sum(torch.isnan(norm_bias).int()) != 0:
                    mask = torch.sum(self.text_bias_features, dim=-1)
                    bias_list = []
                    for m, i, ni in zip(mask, self.text_bias_features, norm_bias):
                        if m == 0:
                            bias_list.append(torch.zeros_like(i))
                        else:
                            bias_list.append(ni)
                    text_bias_features = torch.stack(bias_list)
                    # print('text_bias_features has nan')
                else:
                    text_bias_features = norm_bias
                    # print('text_bias_features has no nan')
            else:
                text_features = self.text_features
                text_bias_features = self.text_bias_features
            ca_text_features = text_features.unsqueeze(dim=0) - self.beta * text_bias_features.unsqueeze(dim=1)
            ca_text_features = ca_text_features / ca_text_features.norm(dim=-1, keepdim=True)
            ca_img_features = ca_img_features.unsqueeze(dim=1)
            ca_logits = self.clip_model.logit_scale.exp() * torch.bmm(ca_img_features, ca_text_features.transpose(1, 2))
            ca_logits = ca_logits.squeeze()

        elif 'img2text_ensembleshift' in self.cfg.TRAINER.CALIBRATE_TEXT:
            if 'norm' in self.cfg.TRAINER.CALIBRATE_TEXT:
                text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
                text_bias_features = self.text_bias_features / self.text_bias_features.norm(dim=-1, keepdim=True)
            else:
                text_features = self.text_features
                text_bias_features = self.text_bias_features
            # print('text_features', text_features.shape, 'text_bias_features', text_bias_features.shape)
            # torch.nn.functional.cosine_similarity(text_features, self.gt_text_features)
            # torch.nn.functional.cosine_similarity(text_features - 2.4 * text_bias_features, self.gt_text_features)
            # import pdb; pdb.set_trace()
            ca_text_features = text_features - self.beta * text_bias_features
            ca_text_features = ca_text_features / ca_text_features.norm(dim=-1, keepdim=True)
            ca_logits = self.clip_model.logit_scale.exp() * ca_img_features @ ca_text_features.t()
        
        elif 'pig3' in self.cfg.TRAINER.CALIBRATE_TEXT:
            if 'norm' in self.cfg.TRAINER.CALIBRATE_TEXT:
                text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
                text_bias_features = self.text_bias_features / self.text_bias_features.norm(dim=-1, keepdim=True)
            else:
                text_features = self.text_features
                text_bias_features = self.text_bias_features
            ca_text_features = text_features - self.beta * text_bias_features
            ca_text_features = ca_text_features / ca_text_features.norm(dim=-1, keepdim=True)
            ca_logits = self.clip_model.logit_scale.exp() * ca_img_features @ ca_text_features.t()

        elif 'pig' in self.cfg.TRAINER.CALIBRATE_TEXT:
            avg_text_features = self.text_features.mean(dim=0)
            if 'norm' in self.cfg.TRAINER.CALIBRATE_TEXT:
                text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True) - avg_text_features/avg_text_features.norm(dim=-1, keepdim=True)
                text_bias_features = self.text_bias_features / self.text_bias_features.norm(dim=-1, keepdim=True)
            else:
                text_features = self.text_features - avg_text_features
                text_bias_features = self.text_bias_features
            ca_text_features = text_features - self.beta * text_bias_features
            ca_text_features = ca_text_features / ca_text_features.norm(dim=-1, keepdim=True)
            ca_logits = self.clip_model.logit_scale.exp() * ca_img_features @ ca_text_features.t()

        elif 'v5' in self.cfg.TRAINER.CALIBRATE_TEXT or 'v6' in self.cfg.TRAINER.CALIBRATE_TEXT:
            ca_text_features = self.img_avg_features.unsqueeze(1) + self.beta * self.text_bias_features     # [100, 1, 512] + [100, 314, 512]
            if 'norm' in self.cfg.TRAINER.CALIBRATE_TEXT:
                ca_text_features = ca_text_features / ca_text_features.norm(dim=-1, keepdim=True)
            ca_img_features = ca_img_features.unsqueeze(dim=1)
            ca_logits = self.clip_model.logit_scale.exp() * torch.bmm(ca_img_features, ca_text_features.transpose(1, 2))
            ca_logits = ca_logits.squeeze()
        
        elif 'feature' in self.cfg.TRAINER.CALIBRATE_TEXT:
            if 'norm' in self.cfg.TRAINER.CALIBRATE_TEXT:
                text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
                text_bias_features = self.text_bias_features / self.text_bias_features.norm(dim=-1, keepdim=True)
            else:
                text_features = self.text_features
                text_bias_features = self.text_bias_features
            ca_text_features = text_features - self.beta * text_bias_features
            ca_text_features = ca_text_features / ca_text_features.norm(dim=-1, keepdim=True)
            ca_logits = self.clip_model.logit_scale.exp() * ca_img_features @ ca_text_features.t()
 
        else:
            ca_text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
            ca_logits = self.clip_model.logit_scale.exp() * ca_img_features @ ca_text_features.t()

        # import pdb; pdb.set_trace()
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
            self.domimg_bias_features_v6 = {}  # for v6
            for i in range(6):    # rebuttal before:上限为6
            # for i in range(10):      # rebuttaling：上限为10
            # for i in range(self.cfg.TRAINER.UNLABELED_CLUSTERS):
                self.domimg_bias_logits[i] = torch.zeros(1,len(classnames), dtype=self.dtype).to('cuda')
                self.domimg_bias_features[i] = torch.tensor([[0]], dtype=self.dtype).to('cuda')
                self.domtext_bias_logits[i] = torch.zeros(len(classnames), dtype=self.dtype).to('cuda')
                # self.domtext_bias_features[i] = torch.tensor([[[0]]], dtype=self.dtype).to('cuda')
                self.domtext_bias_features[i] = torch.tensor([[0]], dtype=self.dtype).to('cuda')
                self.domimg_bias_features_v6[i] = torch.tensor([[0]], dtype=self.dtype).to('cuda')

            
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

            if 'img' in cfg.TRAINER.CALIBRATE_IMG or 'img' in cfg.TRAINER.CALIBRATE_TEXT:
                print(f'calibrating image on domain: {cfg.TRAINER.CALIBRATE_IMG}')
                self.get_image_bias()
                # print('draw bias domain')
                # self.draw_bias_dom(classnames, )

            if 'text' in cfg.TRAINER.CALIBRATE_TEXT:
                print(f'calibrating text on domain: {cfg.TRAINER.CALIBRATE_TEXT}')
                self.get_text_bias()
                # print('draw bias domain')
                # self.draw_bias_dom(classnames)
            
            print('calibration done')
            self.draw_bias_dom(classnames)
                
        # import pdb; pdb.set_trace()
            


    # calibrate名字中含‘img’
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
        loader = self.val_loader
        self.set_model_mode('eval')
        calibrate_type = self.cfg.TRAINER.CALIBRATE_IMG
        keys = []
        need_feature_type = ['img_featlogit', 'img_feature', 'img_feature_main', 'img_feature_main2', 'img_feature_norm', 'img_feature_main_norm', 'img_feature_main2_norm', 'img_featlogit_main', 'img_featlogit_main2', 'img_feature_calibrate', 'img_feature_calibrate_norm', 'img_feature_avg', 'img_feature_avg_norm']
        # need_text_type = ['img2text_shift', 'img2text_shift_norm', 'img2text_ensembleshift', 'img2text_ensembleshift_norm', 'img2text_ensembleshift_norm_v2', 'img2text_shift_norm_v2', 'img2text_shift_modality_norm', 'img2text_ensembleshift_modality_norm']
        for batch_idx, batch in enumerate(tqdm(loader, desc="Collecting domain average information")):
            parsed_data = self.parse_batch_train_dompred(batch)
            # input_u, input_u2, index_u, label_u, domlabel_u = parsed_data
            input_u, index_u, label_u, domlabel_u, domname_list_u = parsed_data

            outputs_u, img_feat = self.get_logits_features(input_u)
            probs_u = torch.softmax(outputs_u, dim=-1)
            max_probs, targets_u = torch.max(probs_u, dim=-1)
            mask = max_probs.ge(self.conf_thre).int()
            # for i, name in enumerate(domname_list_u):    #! domname_list_u is a list of domain names (ground truth)
            # domlabel_u = domlabel_u.cpu()
            for i, name in enumerate(domlabel_u):      #! domname_list_u is a list of domain pred (clustered domain)
                if calibrate_type == 'img_image':
                    domain_img_avg[name].append(input_u[i].cpu())
                elif calibrate_type in need_feature_type:
                    domain_feat_avg[name].append(img_feat[i].cpu())
                elif calibrate_type == 'img_logit':
                    domain_logit_avg[name].append(outputs_u[i].cpu())
                elif calibrate_type == 'img_prob':
                    domain_prob_avg[name].append(probs_u[i].cpu())
                elif 'none' not in self.cfg.TRAINER.CALIBRATE_IMG:
                    raise ValueError(f'Invalid calibration method {self.cfg.TRAINER.CALIBRATE_IMG}')
                
                # if (self.cfg.TRAINER.CALIBRATE_TEXT in need_text_type) and (calibrate_type not in need_feature_type):
                if ('img2text' in self.cfg.TRAINER.CALIBRATE_TEXT) and (calibrate_type not in need_feature_type):
                    domain_feat_avg[name].append(img_feat[i].cpu())
                    
                self.main_domain[name] += mask[i] 
                keys = set(keys).union(set(domlabel_u))
                

        # ! self.main_domain is used for feature calibration/alignment
        self.main_domain = sorted(self.main_domain.keys(), key=lambda item: self.main_domain[item], reverse=True)[0]
        
        print('calibrate img keys: ', keys)
        # if (calibrate_type in need_feature_type) or (self.cfg.TRAINER.CALIBRATE_TEXT in need_text_type):
        if (calibrate_type in need_feature_type) or ('img2text' in self.cfg.TRAINER.CALIBRATE_TEXT):
            for name in list(keys):
                self.domain_feat_avg[name] = torch.stack(domain_feat_avg[name], dim=0).mean(dim=0)  # new precision
                self.domain_feat_avg[name] = self.domain_feat_avg[name].unsqueeze(dim=0)
            self.domain_feat_avg['avg'] = torch.cat([torch.stack(domain_feat_avg[name]) for name in keys], dim=0).mean(dim=0)
            self.domain_feat_avg['avg'] = self.domain_feat_avg['avg'].unsqueeze(dim=0)
            # import pdb; pdb.set_trace()
        print('calibrate main domain: ', self.main_domain)

        if 'pig2' in self.cfg.TRAINER.CALIBRATE_TEXT:
            all_domain_avg_feats = []
            for name in list(keys):
                all_domain_avg_feats.append(self.domain_feat_avg[name])
            all_domain_avg_feats = torch.cat(all_domain_avg_feats, dim=0).cuda()
            all_domain_avg_feats = all_domain_avg_feats / all_domain_avg_feats.norm(dim=-1, keepdim=True)

            text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
            ada_lambda = text_features.mm(all_domain_avg_feats.t())
            ada_lambda = ada_lambda / ada_lambda.sum(dim=-1, keepdim=True)
            # ada_lambda = F.softmax(ada_lambda, dim=-1)
            # check ada_lambda [345, 6]
            # import pdb; pdb.set_trace()

        for name in list(keys):
            if self.cfg.TRAINER.CALIBRATE_IMG == 'img_image':
                # self.domain_img_avg[name] = sum(domain_img_avg[name]) / len(domain_img_avg[name])
                self.domain_img_avg[name] = torch.stack(domain_img_avg[name], dim=0).mean(dim=0)  # new precision
                self.domain_img_avg[name] = self.domain_img_avg[name].unsqueeze(dim=0)
                logits = self.forward(self.domain_img_avg[name].cuda())
                #? self.domain_img_avg[name] = logits
                self.domimg_bias_logits[name] = logits

            elif 'img_featlogit' in self.cfg.TRAINER.CALIBRATE_IMG:
                # 从平均Feature产生的logits角度角度进行校准
                if 'img_featlogit_main' in self.cfg.TRAINER.CALIBRATE_IMG:  #! align with main domain
                    print(f'align with main domain {self.main_domain}')
                    if self.cfg.TRAINER.CALIBRATE_IMG == 'img_featlogit_main2':
                        if name != self.main_domain:
                            self.domain_feat_avg[name] = self.domain_feat_avg[name] - self.domain_feat_avg[self.main_domain]
                    else:
                        self.domain_feat_avg[name] = self.domain_feat_avg[name] - self.domain_feat_avg[self.main_domain]
                # calculate the cosine similarity between the domain feature and the text feature
                self.domain_feat_avg[name] = self.domain_feat_avg[name] / self.domain_feat_avg[name].norm(dim=-1, keepdim=True)
                text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
                logits = self.clip_model.logit_scale.exp() * self.domain_feat_avg[name].cuda() @ text_features.t()
                self.domimg_bias_logits[name] = logits

            elif self.cfg.TRAINER.CALIBRATE_IMG == 'img_logit':
                self.domain_logit_avg[name] = self.calculate_large_tensor(domain_logit_avg[name])
                self.domain_logit_avg[name] = self.domain_logit_avg[name].unsqueeze(dim=0).cuda()
                self.domimg_bias_logits[name] = self.domain_logit_avg[name]

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
                

            elif 'img_prob' in self.cfg.TRAINER.CALIBRATE_IMG:
                # self.domain_prob_avg[name] = sum(domain_prob_avg[name]) / len(domain_prob_avg[name])
                self.domain_prob_avg[name] = torch.stack(domain_prob_avg[name], dim=0).mean(dim=0)  # new precision
                self.domain_prob_avg[name] = self.domain_prob_avg[name].unsqueeze(dim=0).cuda()
 
            if 'img2text' in self.cfg.TRAINER.CALIBRATE_TEXT:
                # 将其他domain的Feature与main domain的Feature作差，作为Feature shift，然后校准text_features
                
                if 'v2' in self.cfg.TRAINER.CALIBRATE_TEXT:   # norm之后进行特征相减
                    # self.domain_feat_avg[name] = self.domain_feat_avg[name].norm(dim=-1, keepdim=True) - self.domain_feat_avg[self.main_domain].norm(dim=-1, keepdim=True)   # old版本
                    self.domain_feat_avg[name] = self.domain_feat_avg[name]/self.domain_feat_avg[name].norm(dim=-1, keepdim=True) - self.domain_feat_avg[self.main_domain]/self.domain_feat_avg[self.main_domain].norm(dim=-1, keepdim=True)
                # elif 'v3' in self.cfg.TRAINER.CALIBRATE_TEXT:
                elif 'v32' in self.cfg.TRAINER.CALIBRATE_TEXT:
                    # 'v32' self.domain_feat_avg['avg']用全部图像，减去全部图像的平均值，而不是减去main domain的平均值
                    self.domain_feat_avg[name] = self.domain_feat_avg[name] - self.domain_feat_avg['avg']
                elif 'v4' in self.cfg.TRAINER.CALIBRATE_TEXT:
                    # self.domain_feat_avg[name] = self.domain_feat_avg[name].norm(dim=-1, keepdim=True) - self.domain_feat_avg['avg'].norm(dim=-1, keepdim=True)   # old版本
                    self.domain_feat_avg[name] = self.domain_feat_avg[name]/self.domain_feat_avg[name].norm(dim=-1, keepdim=True) - self.domain_feat_avg['avg']/self.domain_feat_avg['avg'].norm(dim=-1, keepdim=True)
                elif 'pig' in self.cfg.TRAINER.CALIBRATE_TEXT:   # img2text_pig_ensemble, img2text_pig_ensemble_norm, img2text_pig2_ensemble, img2text_pig2_ensemble_norm
                    self.domain_feat_avg[name] = - self.domain_feat_avg[name] + self.domain_feat_avg['avg']  #! 减去的是avg domain的feature, equal to -'v32'
                    self.domain_feat_avg[name] = self.domain_feat_avg[name] / self.domain_feat_avg[name].norm(dim=-1, keepdim=True)    # normalize it 配合ensemble使用


                # 'img2text_v5_main/avg' 'img2text_v6_main/avg' 'img2text_v6_main/avg_norm' 都是text-avg_img_feature作为bias（和domain无关的，也就是domimg_bias_features_v6中的每个values都是相同的），最终的ca_text_feature是 img_feature[dom] + bias
                elif 'v5' in self.cfg.TRAINER.CALIBRATE_TEXT:  # no normalize
                    self.domimg_bias_features_v6[name] = self.domain_feat_avg[name].cuda()
                    if 'main' in self.cfg.TRAINER.CALIBRATE_TEXT:
                        self.domain_feat_avg[name] = self.text_features - self.domain_feat_avg[self.main_domain]
                    elif 'avg' in self.cfg.TRAINER.CALIBRATE_TEXT:
                        self.domain_feat_avg[name] = self.text_features - self.domain_feat_avg['avg']

                elif 'v6' in self.cfg.TRAINER.CALIBRATE_TEXT:  # after normalize
                    self.domimg_bias_features_v6[name] = self.domain_feat_avg[name].cuda()
                    if 'main' in self.cfg.TRAINER.CALIBRATE_TEXT:
                        self.domain_feat_avg[name] = self.text_features / self.text_features.norm(dim=-1, keepdim=True) - (self.domain_feat_avg['avg'] / self.domain_feat_avg[self.main_domain].norm(dim=-1, keepdim=True)).cuda()
                    elif 'avg' in self.cfg.TRAINER.CALIBRATE_TEXT:
                        self.domain_feat_avg[name] = self.text_features / self.text_features.norm(dim=-1, keepdim=True) - (self.domain_feat_avg['avg'] / self.domain_feat_avg['avg'].norm(dim=-1, keepdim=True)).cuda()
                    # import pdb; pdb.set_trace()

                else:
                    self.domain_feat_avg[name] = self.domain_feat_avg[name] - self.domain_feat_avg[self.main_domain]  #! 减去的是main domain的feature

                self.domtext_bias_features[name] = self.domain_feat_avg[name].cuda()


                #if 'ensemble' in self.cfg.TRAINER.CALIBRATE_TEXT:
                    #self.text_bias_features = sum(self.domtext_bias_features.values()).cuda()
        if 'ensemble' in self.cfg.TRAINER.CALIBRATE_TEXT:
            if 'pig2' in self.cfg.TRAINER.CALIBRATE_TEXT:
                all_direct = []
                for name in list(keys):
                    all_direct.append(self.domtext_bias_features[name])
                all_direct = torch.stack(all_direct, dim=0).cuda()
                all_direct = all_direct.squeeze(dim=1)
                # import pdb; pdb.set_trace()
                # check all direct is [6, dim]
                self.text_bias_features = ada_lambda.mm(all_direct)
            else:
                self.text_bias_features = sum(self.domtext_bias_features.values()).cuda()
                    
        



    def calculate_large_tensor(self, large_list):
        interval = 1000
        interval_list = [sum(large_list[i:i+interval])/interval for i in range(0, len(large_list), interval)]
        weighted_list = sum([len(large_list[i:i+interval])/interval for i in range(0, len(large_list), interval)])
        return sum(interval_list) / weighted_list


    # for text bias，calibrate名字中含‘text’
    def get_text_bias(self):
        self.domain2domlabel = self.dm.dataset.domain2domlabel
        # domain_name = ['clipart', 'infograph','painting', 'quickdraw', 'real', 'sketch']
        domain_name = self.domains
        prompts = domain_name
        if self.cfg.TRAINER.CALIBRATE_TEXT == 'text_logit_domname':
            # ! 一般来讲，这个是最好的，因为这个是最直接的，但是这个需要在训练的时候就知道domain的名字
            prompts = domain_name
        elif self.cfg.TRAINER.CALIBRATE_TEXT == 'text_logit_prompt':
            prompts = [self.temp.format(c.replace(' ', '_')) for c in domain_name]
        elif self.cfg.TRAINER.CALIBRATE_TEXT == 'text_logit_prompt2':
            temp = 'A {} image of a.'
            prompts = [temp.format(c.replace(' ', '_')) for c in domain_name]
        elif 'img2text' in self.cfg.TRAINER.CALIBRATE_TEXT:
            return
        elif 'feature_ensemble' in self.cfg.TRAINER.CALIBRATE_TEXT:
            avg_text_features = torch.mean(self.text_features, dim=0)
            self.text_bias_features = avg_text_features
            return
        elif 'none' not in self.cfg.TRAINER.CALIBRATE_TEXT:
            raise NotImplementedError(f'calibrate text method {self.cfg.TRAINER.CALIBRATE_TEXT} not implemented')

        print('prompts:', prompts)
        prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(prompts)
            norm_text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            clip_text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
            logits = self.clip_model.logit_scale.exp() * norm_text_features @ clip_text_features.t()
            
        for i, name in enumerate(domain_name):
            if 'logit' in self.cfg.TRAINER.CALIBRATE_TEXT:
                self.domtext_bias_logits[self.domain2domlabel[name]] = logits[i].unsqueeze(dim=0)    # ground-truth domain label
            
            # self.text_bias[name] = logits[i].unsqueeze(dim=0)



    def parse_batch_train_dompred(self, batch_u):
        input_u = batch_u["img"]  # weak augmentation
        index_u = batch_u['index']
        label_u = batch_u["label"]
        domlabel_u = batch_u["domlabel"]
        domname_list_u = batch_u["domain"]

        input_u = input_u.to(self.device)
        index_u = index_u.to(self.device)
        label_u = label_u.to(self.device)
        domlabel_u = domlabel_u.numpy()

        return input_u, index_u, label_u, domlabel_u, domname_list_u
    

    # for 'domain' in cfg.TRAINER.CALIBRATE
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
            
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time} seconds")
            return 0   # 仅涉及测试阶段，无需保存current result

        else:     
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                input, label = self.parse_batch_test(batch)
                domlabel = None    # upper bound 
                domain = batch['domain']
                # domlabel = batch['domlabel']    # upper bound 
                # output = self.model_inference(input, domlabel)
                output = self.model_inference(input, domain)
                self.evaluator.process(output, label, len_dom)

            results = self.evaluator.evaluate()

            for k, v in results.items():
                tag = f"{split}/{k}"
                self.write_scalar(tag, v, self.epoch)

            return list(results.values())[0]

        

    def get_features(self, image, domain=None):
        if domain is not None:
            dom_list = [d.item() for d in domain]

            self.image_bias_logits = torch.stack([self.domimg_bias_logits[d] for d in dom_list]).squeeze()
            self.image_bias_features = torch.stack([self.domimg_bias_features[d] for d in dom_list]).squeeze(dim=1)
           
        image_features = self.clip_model.encode_image(image)
        text_features = self.text_features
        image_features_clip = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_clip = text_features / text_features.norm(dim=-1, keepdim=True)

        if 'norm' in self.cfg.TRAINER.CALIBRATE_IMG:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            norm_bias = self.image_bias_features / self.image_bias_features.norm(dim=-1, keepdim=True)
            if torch.sum(torch.isnan(norm_bias).int()) != 0:
                mask = torch.sum(self.image_bias_features, dim=-1)
                bias_list = []
                for m, i, ni in zip(mask, self.image_bias_features, norm_bias):
                    if m == 0:
                        bias_list.append(torch.zeros_like(i))
                    else:
                        bias_list.append(ni)
                self.image_bias_features = torch.stack(bias_list)
                # import pdb; pdb.set_trace()
            else:
                self.image_bias_features = norm_bias

        # calibrate features
        ca_img_features = image_features - self.alpha * self.image_bias_features
        
        print(image_features_clip.shape, ca_img_features.shape)
        # import pdb; pdb.set_trace()
        return image_features_clip, ca_img_features

    def get_textfeatures(self):
        # dom_list = self.domtext_bias_features.keys()
        dom_list = list(range(len(self.domtext_bias_features)))
        if 'ensemble' not in self.cfg.TRAINER.CALIBRATE_TEXT:    
            self.text_bias_features = torch.stack([self.domtext_bias_features[d] for d in dom_list]).squeeze(dim=1)
        if 'v6' in self.cfg.TRAINER.CALIBRATE_TEXT:
            if 'ensemimg' in self.cfg.TRAINER.CALIBRATE_TEXT:   #! ensemble image features 是没有意义的，相当于再重新加上bias中减去的平均
                self.img_avg_features = torch.mean(torch.stack(list(self.domimg_bias_features_v6.values())),dim=0)
            else:
                self.img_avg_features = torch.stack([self.domimg_bias_features_v6[d] for d in dom_list]).squeeze(dim=1)

        ca_text_features = None
        if 'img2text_shift' in self.cfg.TRAINER.CALIBRATE_TEXT:
            if 'norm' in self.cfg.TRAINER.CALIBRATE_TEXT:
                text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
                norm_bias = self.text_bias_features / self.text_bias_features.norm(dim=-1, keepdim=True)
                if torch.sum(torch.isnan(norm_bias).int()) != 0:
                    mask = torch.sum(self.text_bias_features, dim=-1)
                    bias_list = []
                    for m, i, ni in zip(mask, self.text_bias_features, norm_bias):
                        if m == 0:
                            bias_list.append(torch.zeros_like(i))
                        else:
                            bias_list.append(ni)
                    text_bias_features = torch.stack(bias_list)
                else:
                    text_bias_features = norm_bias
            else:
                text_features = self.text_features
                text_bias_features = self.text_bias_features
            ca_text_features = text_features.unsqueeze(dim=0) - self.beta * text_bias_features.unsqueeze(dim=1)
            ca_text_features = ca_text_features / ca_text_features.norm(dim=-1, keepdim=True)
            ca_text_features = ca_text_features.squeeze(dim=0)
        elif 'img2text_ensembleshift' in self.cfg.TRAINER.CALIBRATE_TEXT:
            if 'norm' in self.cfg.TRAINER.CALIBRATE_TEXT:
                text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
                text_bias_features = self.text_bias_features / self.text_bias_features.norm(dim=-1, keepdim=True)
            else:
                text_features = self.text_features
                text_bias_features = self.text_bias_features
            ca_text_features = text_features.unsqueeze(dim=0) - self.beta * text_bias_features.unsqueeze(dim=1)
            ca_text_features = ca_text_features / ca_text_features.norm(dim=-1, keepdim=True)
            ca_text_features = ca_text_features.squeeze(dim=0)

        elif 'pig3' in self.cfg.TRAINER.CALIBRATE_TEXT:
            if 'norm' in self.cfg.TRAINER.CALIBRATE_TEXT:
                text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
                text_bias_features = self.text_bias_features / self.text_bias_features.norm(dim=-1, keepdim=True)
            else:
                text_features = self.text_features
                text_bias_features = self.text_bias_features
            ca_text_features = text_features - self.beta * text_bias_features
            # import pdb; pdb.set_trace()
            # ca_text_features = ca_text_features / ca_text_features.norm(dim=-1, keepdim=True)

        elif 'v5' in self.cfg.TRAINER.CALIBRATE_TEXT or 'v6' in self.cfg.TRAINER.CALIBRATE_TEXT:
            ca_text_features = self.img_avg_features.unsqueeze(1) + self.beta * self.text_bias_features     # [100, 1, 512] + [100, 314, 512]
            if 'norm' in self.cfg.TRAINER.CALIBRATE_TEXT:
                ca_text_features = ca_text_features / ca_text_features.norm(dim=-1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        return text_features, ca_text_features

    @torch.no_grad()
    def get_features_test(self, dataloader=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        vt_fea = defaultdict(list)
        if dataloader is None:
            data_loader = self.eval_train_loader_u
        elif dataloader == 'val':
            data_loader = self.val_loader
            
        for _, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            dom_label = batch["domlabel"]
            dom_label = dom_label.to(self.device)
            domain = batch["domain"]
            impath = batch["impath"]
            # get features
            if hasattr(self, 'model'):
                v_fea, v_fea_ttc = self.model.get_features(input, dom_label)
            elif hasattr(self, 'get_features'):
                v_fea, v_fea_ttc = self.get_features(input, dom_label)
            vt_fea['v_fea'].extend(v_fea)
            # vt_fea['t_fea'].extend(t_fea)
            vt_fea['label'].extend(label)
            vt_fea['v_fea_ttc']. extend(v_fea_ttc)
            # vt_fea['t_fea_ttc']. extend(t_fea_ttc)
            vt_fea['dom_label'].extend(dom_label)
            vt_fea['domain'].extend(domain)
            vt_fea['impath'].extend(impath)
        
        t_fea, t_fea_ttc = self.get_textfeatures()
        vt_fea['t_fea'].extend(t_fea)
        vt_fea['t_fea_ttc']. extend(t_fea_ttc)

        vt_fea['classnames'] = self.dm.dataset.classnames

        return vt_fea



    def draw_bias(self, classnames, name=-1, type_bias='image'):
        import matplotlib.pyplot as plt
        if self.image_bias_probs.dim() > 1:
            self.image_bias_probs = self.image_bias_probs[0]
        probs, lab = torch.topk(self.image_bias_probs, len(classnames))
        clab = [classnames[i] for i in lab]
        uniform_probs = torch.ones(len(classnames)) / len(classnames)
        # plt.figure(figsize=(15, 20), dpi=500)   # 3 figures
        plt.figure(figsize=(15, 10), dpi=500)      # 2 figures
        plt.tight_layout()
        ax1 = plt.subplot(2, 1, 1)
        ax1.bar(clab, probs.cpu().numpy(), label='bias')
        ax1.plot(clab, uniform_probs.cpu().numpy(), 'r--', label='uniform')
        ax1.tick_params()
        plt.xticks([])
        ax1.set_xlabel('class')
        ax1.set_ylabel('prob')
        ax1.set_title('Text bias probs')
        ax1.legend()

        topk = torch.sum(self.image_bias_probs.ge(1/len(classnames)))
        probs, lab = torch.topk(self.image_bias_probs, topk.item())
        clab = [classnames[i] for i in lab]
        ax2 = plt.subplot(2, 1, 2)
        ax2.bar(clab, probs.cpu().numpy(), label='bias')
        ax2.plot(clab, uniform_probs.cpu().numpy()[:topk], 'r--', label='uniform')
        # ax2.tick_params(axis='x', rotation=45, fontsize=5)
        plt.xticks(fontsize=6, rotation=45)
        ax2.legend()
        # ax2.set_xlabel('class')
        ax2.set_ylabel('prob')
        ax2.set_title(f'top{topk} text bias probs greater than uniform probs')
       
        plt.savefig(os.path.join(self.cfg.OUTPUT_DIR, type_bias, f'{name}_bias_probs.png'))
        plt.close()
        plt.clf()


    def draw_bias_dom(self, classnames):
        # draw_dict = None
        draw_dict = {}
        type_bias = ''
        if self.cfg.TRAINER.CALIBRATE_IMG in ['img_image', 'img_featlogit', 'img_logit']:
        # if self.cfg.TRAINER.CALIBRATE_IMG in ['img_image', 'img_featlogit', 'img_logit', 'img_feature', 'img_feature_norm']:
            print('draw domimg bias')
            draw_dict = self.domimg_bias_logits
            type_bias = 'image_bias_figs'
        if self.cfg.TRAINER.CALIBRATE_TEXT in ['text_logit_domname', 'text_logit_prompt', 'text_logit_prompt2']:
        # elif self.cfg.TRAINER.CALIBRATE_TEXT in ['text_logit_prompt', 'text_logit_prompt2', 'text_feature', 'text_feature_norm']:
            print('draw text bias')
            draw_dict = self.domtext_bias_logits
            type_bias = 'text_bias_figs'
        mkdir_if_missing(os.path.join(self.cfg.OUTPUT_DIR, type_bias))
        for name in draw_dict.keys():
            self.image_bias_probs = torch.softmax(draw_dict[name], dim=-1)
            # import pdb; pdb.set_trace()
            self.draw_bias(classnames, name, type_bias)

@TRAINER_REGISTRY.register()
class ZeroshotCLIP_calibrate_v3_ensemble(ZeroshotCLIP_calibrate_v3):
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
        # mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model


        # # ground truth emsembel text embedding
        # num_temp = len(self.templates)
        # print(f"ground-truth Prompt ensembling (n={num_temp})")
        # print('templates: ', self.templates)

        # mean_text_features = 0
        # for i, temp in enumerate(self.templates):
        #     prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        #     prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        #     text_features = clip_model.encode_text(prompts)
        #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #     mean_text_features = mean_text_features + text_features.detach()
        # mean_text_features = mean_text_features / num_temp
        # mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)
        # self.gt_text_features = mean_text_features

        self.dtype = clip_model.dtype
        self.T = 1.0
        self.conf_thre = 0.95
        self.alpha = cfg.TRAINER.CALIBRATE_IMG_WEIGHT
        self.beta = cfg.TRAINER.CALIBRATE_TEXT_WEIGHT

        import time
        start_time = time.time()  # 记录开始时间

        self.calibrate(cfg, classnames)
        
        end_time = time.time()  # 记录结束时间
        execution_time = end_time - start_time  # 计算执行时间
        print(f"Execution time: {execution_time} seconds")
        

# ? 这个和ensemble的区别就是，Multi用的是domain prompt集成，ensemble用的是imagenet template集成
@TRAINER_REGISTRY.register()
class ZeroshotCLIP_calibrate_v3_multi(ZeroshotCLIP_calibrate_v3):
    templates = DOMAINNET_TEMPLATES

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
        self.clip_model = clip_model

        # emsemble text embedding
        num_temp = len(self.templates)
        print(f"ground-truth Prompt ensembling (n={num_temp})")
        print('templates: ', self.templates)

        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features.detach()
        mean_text_features = mean_text_features / num_temp
        # mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)
        self.text_features = mean_text_features

        self.dtype = clip_model.dtype
        self.T = 1.0
        self.conf_thre = 0.95
        self.alpha = cfg.TRAINER.CALIBRATE_IMG_WEIGHT
        self.beta = cfg.TRAINER.CALIBRATE_TEXT_WEIGHT
        self.calibrate(cfg, classnames)
     

@TRAINER_REGISTRY.register()
class ZeroshotCLIP_calibrate_v3_summary(ZeroshotCLIP_calibrate_v3):
    
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
            alpha_list = np.arange(0.1, 1.2, 0.1)
            beta_list = np.arange(0.1, 1.0, 0.1)

            for alpha in alpha_list:
                self.alpha = alpha
                for beta in beta_list:
                    self.beta = beta

                    for domain, loader in data_loader.items():
                        print(f'Test Accuracy on {domain}' )
                        cur_evaluator.reset()
                        count = 0
                        all_count = 0
                        for batch_idx, batch in enumerate(tqdm(loader)):
                            input, label = self.parse_batch_test(batch)
                            domlabel = batch['domlabel']    #! domain pred: cluster name
                            count += sum(label ==65)
                            all_count += label.shape[0]
            
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

            return 0   # 仅涉及测试阶段，无需保存current result



@TRAINER_REGISTRY.register()
class ZeroshotCLIP_calibrate_v3_trainu_summary(ZeroshotCLIP_calibrate_v3):
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
                        count = 0
                        all_count = 0
                        for batch_idx, batch in enumerate(tqdm(loader)):
                            input, label = self.parse_batch_test(batch)
                            domlabel = batch['domlabel']    #! domain pred: cluster name
                            count += sum(label ==65)
                            all_count += label.shape[0]
            
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

            return 0   # 仅涉及测试阶段，无需保存current result


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
        #! loader = self.val_loader   
        loader = self.eval_train_loader_u   #! 替换为利用训练集的数据得到校准细腻系
        self.set_model_mode('eval')
        calibrate_type = self.cfg.TRAINER.CALIBRATE_IMG
        keys = []
        need_feature_type = ['img_featlogit', 'img_feature', 'img_feature_main', 'img_feature_main2', 'img_feature_norm', 'img_feature_main_norm', 'img_feature_main2_norm', 'img_featlogit_main', 'img_featlogit_main2', 'img_feature_calibrate', 'img_feature_calibrate_norm', 'img_feature_avg', 'img_feature_avg_norm']
        # need_text_type = ['img2text_shift', 'img2text_shift_norm', 'img2text_ensembleshift', 'img2text_ensembleshift_norm', 'img2text_ensembleshift_norm_v2', 'img2text_shift_norm_v2', 'img2text_shift_modality_norm', 'img2text_ensembleshift_modality_norm']
        for batch_idx, batch in enumerate(tqdm(loader, desc="Collecting domain average information")):
            parsed_data = self.parse_batch_train_dompred(batch)
            # input_u, input_u2, index_u, label_u, domlabel_u = parsed_data
            input_u, index_u, label_u, domlabel_u, domname_list_u = parsed_data

            outputs_u, img_feat = self.get_logits_features(input_u)
            probs_u = torch.softmax(outputs_u, dim=-1)
            max_probs, targets_u = torch.max(probs_u, dim=-1)
            mask = max_probs.ge(self.conf_thre).int()
            # for i, name in enumerate(domname_list_u):    #! domname_list_u is a list of domain names (ground truth)
            # domlabel_u = domlabel_u.cpu()
            for i, name in enumerate(domlabel_u):      #! domname_list_u is a list of domain pred (clustered domain)
                if calibrate_type == 'img_image':
                    domain_img_avg[name].append(input_u[i].cpu())
                elif calibrate_type in need_feature_type:
                    domain_feat_avg[name].append(img_feat[i].cpu())
                elif calibrate_type == 'img_logit':
                    domain_logit_avg[name].append(outputs_u[i].cpu())
                elif calibrate_type == 'img_prob':
                    domain_prob_avg[name].append(probs_u[i].cpu())
                elif 'none' not in self.cfg.TRAINER.CALIBRATE_IMG:
                    raise ValueError(f'Invalid calibration method {self.cfg.TRAINER.CALIBRATE_IMG}')
                
                # if (self.cfg.TRAINER.CALIBRATE_TEXT in need_text_type) and (calibrate_type not in need_feature_type):
                if ('img2text' in self.cfg.TRAINER.CALIBRATE_TEXT) and (calibrate_type not in need_feature_type):
                    domain_feat_avg[name].append(img_feat[i].cpu())
                    
                self.main_domain[name] += mask[i] 
                keys = set(keys).union(set(domlabel_u))
                

        # ! self.main_domain is used for feature calibration/alignment
        self.main_domain = sorted(self.main_domain.keys(), key=lambda item: self.main_domain[item], reverse=True)[0]
        
        print('calibrate img keys: ', keys)
        # if (calibrate_type in need_feature_type) or (self.cfg.TRAINER.CALIBRATE_TEXT in need_text_type):
        if self.cfg.TRAINER.UNLABELED_CLUSTERS > len(list(keys)):
            keys = list(range(self.cfg.TRAINER.UNLABELED_CLUSTERS))
        if (calibrate_type in need_feature_type) or ('img2text' in self.cfg.TRAINER.CALIBRATE_TEXT):
            for name in list(keys):
                self.domain_feat_avg[name] = torch.stack(domain_feat_avg[name], dim=0).mean(dim=0)  # new precision
                self.domain_feat_avg[name] = self.domain_feat_avg[name].unsqueeze(dim=0)
            self.domain_feat_avg['avg'] = torch.cat([torch.stack(domain_feat_avg[name]) for name in keys], dim=0).mean(dim=0)
            self.domain_feat_avg['avg'] = self.domain_feat_avg['avg'].unsqueeze(dim=0)
            # import pdb; pdb.set_trace()
        print('calibrate main domain: ', self.main_domain)

        if 'pig2' in self.cfg.TRAINER.CALIBRATE_TEXT:
            all_domain_avg_feats = []
            for name in list(keys):
                all_domain_avg_feats.append(self.domain_feat_avg[name])
            all_domain_avg_feats = torch.cat(all_domain_avg_feats, dim=0).cuda()
            all_domain_avg_feats = all_domain_avg_feats / all_domain_avg_feats.norm(dim=-1, keepdim=True)

            text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
            ada_lambda = text_features.mm(all_domain_avg_feats.t())
            ada_lambda = ada_lambda / ada_lambda.sum(dim=-1, keepdim=True)
            # ada_lambda = F.softmax(ada_lambda, dim=-1)
            # check ada_lambda [345, 6]
            # import pdb; pdb.set_trace()

        for name in list(keys):
            if self.cfg.TRAINER.CALIBRATE_IMG == 'img_image':
                # self.domain_img_avg[name] = sum(domain_img_avg[name]) / len(domain_img_avg[name])
                self.domain_img_avg[name] = torch.stack(domain_img_avg[name], dim=0).mean(dim=0)  # new precision
                self.domain_img_avg[name] = self.domain_img_avg[name].unsqueeze(dim=0)
                logits = self.forward(self.domain_img_avg[name].cuda())
                #? self.domain_img_avg[name] = logits
                self.domimg_bias_logits[name] = logits

            elif 'img_featlogit' in self.cfg.TRAINER.CALIBRATE_IMG:
                # 从平均Feature产生的logits角度角度进行校准
                if 'img_featlogit_main' in self.cfg.TRAINER.CALIBRATE_IMG:  #! align with main domain
                    print(f'align with main domain {self.main_domain}')
                    if self.cfg.TRAINER.CALIBRATE_IMG == 'img_featlogit_main2':
                        if name != self.main_domain:
                            self.domain_feat_avg[name] = self.domain_feat_avg[name] - self.domain_feat_avg[self.main_domain]
                    else:
                        self.domain_feat_avg[name] = self.domain_feat_avg[name] - self.domain_feat_avg[self.main_domain]
                # calculate the cosine similarity between the domain feature and the text feature
                self.domain_feat_avg[name] = self.domain_feat_avg[name] / self.domain_feat_avg[name].norm(dim=-1, keepdim=True)
                text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
                logits = self.clip_model.logit_scale.exp() * self.domain_feat_avg[name].cuda() @ text_features.t()
                self.domimg_bias_logits[name] = logits

            elif self.cfg.TRAINER.CALIBRATE_IMG == 'img_logit':
                self.domain_logit_avg[name] = self.calculate_large_tensor(domain_logit_avg[name])
                self.domain_logit_avg[name] = self.domain_logit_avg[name].unsqueeze(dim=0).cuda()
                self.domimg_bias_logits[name] = self.domain_logit_avg[name]

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
                

            elif 'img_prob' in self.cfg.TRAINER.CALIBRATE_IMG:
                # self.domain_prob_avg[name] = sum(domain_prob_avg[name]) / len(domain_prob_avg[name])
                self.domain_prob_avg[name] = torch.stack(domain_prob_avg[name], dim=0).mean(dim=0)  # new precision
                self.domain_prob_avg[name] = self.domain_prob_avg[name].unsqueeze(dim=0).cuda()
 
            if 'img2text' in self.cfg.TRAINER.CALIBRATE_TEXT:
                # 将其他domain的Feature与main domain的Feature作差，作为Feature shift，然后校准text_features
                
                if 'v2' in self.cfg.TRAINER.CALIBRATE_TEXT:   # norm之后进行特征相减
                    # self.domain_feat_avg[name] = self.domain_feat_avg[name].norm(dim=-1, keepdim=True) - self.domain_feat_avg[self.main_domain].norm(dim=-1, keepdim=True)   # old版本
                    self.domain_feat_avg[name] = self.domain_feat_avg[name]/self.domain_feat_avg[name].norm(dim=-1, keepdim=True) - self.domain_feat_avg[self.main_domain]/self.domain_feat_avg[self.main_domain].norm(dim=-1, keepdim=True)
                # elif 'v3' in self.cfg.TRAINER.CALIBRATE_TEXT:
                elif 'v32' in self.cfg.TRAINER.CALIBRATE_TEXT:
                    # 'v32' self.domain_feat_avg['avg']用全部图像，减去全部图像的平均值，而不是减去main domain的平均值
                    self.domain_feat_avg[name] = self.domain_feat_avg[name] - self.domain_feat_avg['avg']
                elif 'v4' in self.cfg.TRAINER.CALIBRATE_TEXT:
                    # self.domain_feat_avg[name] = self.domain_feat_avg[name].norm(dim=-1, keepdim=True) - self.domain_feat_avg['avg'].norm(dim=-1, keepdim=True)   # old版本
                    self.domain_feat_avg[name] = self.domain_feat_avg[name]/self.domain_feat_avg[name].norm(dim=-1, keepdim=True) - self.domain_feat_avg['avg']/self.domain_feat_avg['avg'].norm(dim=-1, keepdim=True)
                elif 'pig' in self.cfg.TRAINER.CALIBRATE_TEXT:   # img2text_pig_ensemble, img2text_pig_ensemble_norm, img2text_pig2_ensemble, img2text_pig2_ensemble_norm
                    self.domain_feat_avg[name] = - self.domain_feat_avg[name] + self.domain_feat_avg['avg']  #! 减去的是avg domain的feature, equal to -'v32'
                    self.domain_feat_avg[name] = self.domain_feat_avg[name] / self.domain_feat_avg[name].norm(dim=-1, keepdim=True)    # normalize it 配合ensemble使用


                # 'img2text_v5_main/avg' 'img2text_v6_main/avg' 'img2text_v6_main/avg_norm' 都是text-avg_img_feature作为bias（和domain无关的，也就是domimg_bias_features_v6中的每个values都是相同的），最终的ca_text_feature是 img_feature[dom] + bias
                elif 'v5' in self.cfg.TRAINER.CALIBRATE_TEXT:  # no normalize
                    self.domimg_bias_features_v6[name] = self.domain_feat_avg[name].cuda()
                    if 'main' in self.cfg.TRAINER.CALIBRATE_TEXT:
                        self.domain_feat_avg[name] = self.text_features - self.domain_feat_avg[self.main_domain]
                    elif 'avg' in self.cfg.TRAINER.CALIBRATE_TEXT:
                        self.domain_feat_avg[name] = self.text_features - self.domain_feat_avg['avg']

                elif 'v6' in self.cfg.TRAINER.CALIBRATE_TEXT:  # after normalize
                    self.domimg_bias_features_v6[name] = self.domain_feat_avg[name].cuda()
                    if 'main' in self.cfg.TRAINER.CALIBRATE_TEXT:
                        self.domain_feat_avg[name] = self.text_features / self.text_features.norm(dim=-1, keepdim=True) - (self.domain_feat_avg['avg'] / self.domain_feat_avg[self.main_domain].norm(dim=-1, keepdim=True)).cuda()
                    elif 'avg' in self.cfg.TRAINER.CALIBRATE_TEXT:
                        self.domain_feat_avg[name] = self.text_features / self.text_features.norm(dim=-1, keepdim=True) - (self.domain_feat_avg['avg'] / self.domain_feat_avg['avg'].norm(dim=-1, keepdim=True)).cuda()
                    # import pdb; pdb.set_trace()

                else:
                    self.domain_feat_avg[name] = self.domain_feat_avg[name] - self.domain_feat_avg[self.main_domain]  #! 减去的是main domain的feature

                self.domtext_bias_features[name] = self.domain_feat_avg[name].cuda()

        if 'ensemble' in self.cfg.TRAINER.CALIBRATE_TEXT:
            if 'pig2' in self.cfg.TRAINER.CALIBRATE_TEXT:
                all_direct = []
                for name in list(keys):
                    all_direct.append(self.domtext_bias_features[name])
                all_direct = torch.stack(all_direct, dim=0).cuda()
                all_direct = all_direct.squeeze(dim=1)
                # import pdb; pdb.set_trace()
                # check all direct is [6, dim]
                self.text_bias_features = ada_lambda.mm(all_direct)
            else:
                self.text_bias_features = sum(self.domtext_bias_features.values()).cuda()
                    
  

@TRAINER_REGISTRY.register()
class ZeroshotCLIP_calibrate_v3_ensemble_trainu_summary(ZeroshotCLIP_calibrate_v3_ensemble):
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
                        count = 0
                        all_count = 0
                        for batch_idx, batch in enumerate(tqdm(loader)):
                            input, label = self.parse_batch_test(batch)
                            domlabel = batch['domlabel']    #! domain pred: cluster name
                            count += sum(label ==65)
                            all_count += label.shape[0]
            
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

            return 0   # 仅涉及测试阶段，无需保存current result


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
        #! loader = self.val_loader   
        loader = self.eval_train_loader_u   #! 替换为利用训练集的数据得到校准细腻系
        self.set_model_mode('eval')
        calibrate_type = self.cfg.TRAINER.CALIBRATE_IMG
        keys = []
        need_feature_type = ['img_featlogit', 'img_feature', 'img_feature_main', 'img_feature_main2', 'img_feature_norm', 'img_feature_main_norm', 'img_feature_main2_norm', 'img_featlogit_main', 'img_featlogit_main2', 'img_feature_calibrate', 'img_feature_calibrate_norm', 'img_feature_avg', 'img_feature_avg_norm']
        # need_text_type = ['img2text_shift', 'img2text_shift_norm', 'img2text_ensembleshift', 'img2text_ensembleshift_norm', 'img2text_ensembleshift_norm_v2', 'img2text_shift_norm_v2', 'img2text_shift_modality_norm', 'img2text_ensembleshift_modality_norm']
        for batch_idx, batch in enumerate(tqdm(loader, desc="Collecting domain average information")):
            parsed_data = self.parse_batch_train_dompred(batch)
            # input_u, input_u2, index_u, label_u, domlabel_u = parsed_data
            input_u, index_u, label_u, domlabel_u, domname_list_u = parsed_data

            outputs_u, img_feat = self.get_logits_features(input_u)
            probs_u = torch.softmax(outputs_u, dim=-1)
            max_probs, targets_u = torch.max(probs_u, dim=-1)
            mask = max_probs.ge(self.conf_thre).int()
            # for i, name in enumerate(domname_list_u):    #! domname_list_u is a list of domain names (ground truth)
            # domlabel_u = domlabel_u.cpu()
            for i, name in enumerate(domlabel_u):      #! domname_list_u is a list of domain pred (clustered domain)
                if calibrate_type == 'img_image':
                    domain_img_avg[name].append(input_u[i].cpu())
                elif calibrate_type in need_feature_type:
                    domain_feat_avg[name].append(img_feat[i].cpu())
                elif calibrate_type == 'img_logit':
                    domain_logit_avg[name].append(outputs_u[i].cpu())
                elif calibrate_type == 'img_prob':
                    domain_prob_avg[name].append(probs_u[i].cpu())
                elif 'none' not in self.cfg.TRAINER.CALIBRATE_IMG:
                    raise ValueError(f'Invalid calibration method {self.cfg.TRAINER.CALIBRATE_IMG}')
                
                # if (self.cfg.TRAINER.CALIBRATE_TEXT in need_text_type) and (calibrate_type not in need_feature_type):
                if ('img2text' in self.cfg.TRAINER.CALIBRATE_TEXT) and (calibrate_type not in need_feature_type):
                    domain_feat_avg[name].append(img_feat[i].cpu())
                    
                self.main_domain[name] += mask[i] 
                keys = set(keys).union(set(domlabel_u))
                

        # ! self.main_domain is used for feature calibration/alignment
        self.main_domain = sorted(self.main_domain.keys(), key=lambda item: self.main_domain[item], reverse=True)[0]
        
        print('calibrate img keys: ', keys)
        # if (calibrate_type in need_feature_type) or (self.cfg.TRAINER.CALIBRATE_TEXT in need_text_type):
        if (calibrate_type in need_feature_type) or ('img2text' in self.cfg.TRAINER.CALIBRATE_TEXT):
            for name in list(keys):
                self.domain_feat_avg[name] = torch.stack(domain_feat_avg[name], dim=0).mean(dim=0)  # new precision
                self.domain_feat_avg[name] = self.domain_feat_avg[name].unsqueeze(dim=0)
            self.domain_feat_avg['avg'] = torch.cat([torch.stack(domain_feat_avg[name]) for name in keys], dim=0).mean(dim=0)
            self.domain_feat_avg['avg'] = self.domain_feat_avg['avg'].unsqueeze(dim=0)
            # import pdb; pdb.set_trace()
        print('calibrate main domain: ', self.main_domain)

        if 'pig2' in self.cfg.TRAINER.CALIBRATE_TEXT:
            all_domain_avg_feats = []
            for name in list(keys):
                all_domain_avg_feats.append(self.domain_feat_avg[name])
            all_domain_avg_feats = torch.cat(all_domain_avg_feats, dim=0).cuda()
            all_domain_avg_feats = all_domain_avg_feats / all_domain_avg_feats.norm(dim=-1, keepdim=True)

            text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
            ada_lambda = text_features.mm(all_domain_avg_feats.t())
            ada_lambda = ada_lambda / ada_lambda.sum(dim=-1, keepdim=True)
            # ada_lambda = F.softmax(ada_lambda, dim=-1)
            # check ada_lambda [345, 6]
            # import pdb; pdb.set_trace()

        for name in list(keys):
            if self.cfg.TRAINER.CALIBRATE_IMG == 'img_image':
                # self.domain_img_avg[name] = sum(domain_img_avg[name]) / len(domain_img_avg[name])
                self.domain_img_avg[name] = torch.stack(domain_img_avg[name], dim=0).mean(dim=0)  # new precision
                self.domain_img_avg[name] = self.domain_img_avg[name].unsqueeze(dim=0)
                logits = self.forward(self.domain_img_avg[name].cuda())
                #? self.domain_img_avg[name] = logits
                self.domimg_bias_logits[name] = logits

            elif 'img_featlogit' in self.cfg.TRAINER.CALIBRATE_IMG:
                # 从平均Feature产生的logits角度角度进行校准
                if 'img_featlogit_main' in self.cfg.TRAINER.CALIBRATE_IMG:  #! align with main domain
                    print(f'align with main domain {self.main_domain}')
                    if self.cfg.TRAINER.CALIBRATE_IMG == 'img_featlogit_main2':
                        if name != self.main_domain:
                            self.domain_feat_avg[name] = self.domain_feat_avg[name] - self.domain_feat_avg[self.main_domain]
                    else:
                        self.domain_feat_avg[name] = self.domain_feat_avg[name] - self.domain_feat_avg[self.main_domain]
                # calculate the cosine similarity between the domain feature and the text feature
                self.domain_feat_avg[name] = self.domain_feat_avg[name] / self.domain_feat_avg[name].norm(dim=-1, keepdim=True)
                text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
                logits = self.clip_model.logit_scale.exp() * self.domain_feat_avg[name].cuda() @ text_features.t()
                self.domimg_bias_logits[name] = logits

            elif self.cfg.TRAINER.CALIBRATE_IMG == 'img_logit':
                self.domain_logit_avg[name] = self.calculate_large_tensor(domain_logit_avg[name])
                self.domain_logit_avg[name] = self.domain_logit_avg[name].unsqueeze(dim=0).cuda()
                self.domimg_bias_logits[name] = self.domain_logit_avg[name]

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
                

            elif 'img_prob' in self.cfg.TRAINER.CALIBRATE_IMG:
                # self.domain_prob_avg[name] = sum(domain_prob_avg[name]) / len(domain_prob_avg[name])
                self.domain_prob_avg[name] = torch.stack(domain_prob_avg[name], dim=0).mean(dim=0)  # new precision
                self.domain_prob_avg[name] = self.domain_prob_avg[name].unsqueeze(dim=0).cuda()
 
            if 'img2text' in self.cfg.TRAINER.CALIBRATE_TEXT:
                # 将其他domain的Feature与main domain的Feature作差，作为Feature shift，然后校准text_features
                
                if 'v2' in self.cfg.TRAINER.CALIBRATE_TEXT:   # norm之后进行特征相减
                    # self.domain_feat_avg[name] = self.domain_feat_avg[name].norm(dim=-1, keepdim=True) - self.domain_feat_avg[self.main_domain].norm(dim=-1, keepdim=True)   # old版本
                    self.domain_feat_avg[name] = self.domain_feat_avg[name]/self.domain_feat_avg[name].norm(dim=-1, keepdim=True) - self.domain_feat_avg[self.main_domain]/self.domain_feat_avg[self.main_domain].norm(dim=-1, keepdim=True)
                # elif 'v3' in self.cfg.TRAINER.CALIBRATE_TEXT:
                elif 'v32' in self.cfg.TRAINER.CALIBRATE_TEXT:
                    # 'v32' self.domain_feat_avg['avg']用全部图像，减去全部图像的平均值，而不是减去main domain的平均值
                    self.domain_feat_avg[name] = self.domain_feat_avg[name] - self.domain_feat_avg['avg']
                elif 'v4' in self.cfg.TRAINER.CALIBRATE_TEXT:
                    # self.domain_feat_avg[name] = self.domain_feat_avg[name].norm(dim=-1, keepdim=True) - self.domain_feat_avg['avg'].norm(dim=-1, keepdim=True)   # old版本
                    self.domain_feat_avg[name] = self.domain_feat_avg[name]/self.domain_feat_avg[name].norm(dim=-1, keepdim=True) - self.domain_feat_avg['avg']/self.domain_feat_avg['avg'].norm(dim=-1, keepdim=True)
                elif 'pig' in self.cfg.TRAINER.CALIBRATE_TEXT:   # img2text_pig_ensemble, img2text_pig_ensemble_norm, img2text_pig2_ensemble, img2text_pig2_ensemble_norm
                    self.domain_feat_avg[name] = - self.domain_feat_avg[name] + self.domain_feat_avg['avg']  #! 减去的是avg domain的feature, equal to -'v32'
                    self.domain_feat_avg[name] = self.domain_feat_avg[name] / self.domain_feat_avg[name].norm(dim=-1, keepdim=True)    # normalize it 配合ensemble使用


                # 'img2text_v5_main/avg' 'img2text_v6_main/avg' 'img2text_v6_main/avg_norm' 都是text-avg_img_feature作为bias（和domain无关的，也就是domimg_bias_features_v6中的每个values都是相同的），最终的ca_text_feature是 img_feature[dom] + bias
                elif 'v5' in self.cfg.TRAINER.CALIBRATE_TEXT:  # no normalize
                    self.domimg_bias_features_v6[name] = self.domain_feat_avg[name].cuda()
                    if 'main' in self.cfg.TRAINER.CALIBRATE_TEXT:
                        self.domain_feat_avg[name] = self.text_features - self.domain_feat_avg[self.main_domain]
                    elif 'avg' in self.cfg.TRAINER.CALIBRATE_TEXT:
                        self.domain_feat_avg[name] = self.text_features - self.domain_feat_avg['avg']

                elif 'v6' in self.cfg.TRAINER.CALIBRATE_TEXT:  # after normalize
                    self.domimg_bias_features_v6[name] = self.domain_feat_avg[name].cuda()
                    if 'main' in self.cfg.TRAINER.CALIBRATE_TEXT:
                        self.domain_feat_avg[name] = self.text_features / self.text_features.norm(dim=-1, keepdim=True) - (self.domain_feat_avg['avg'] / self.domain_feat_avg[self.main_domain].norm(dim=-1, keepdim=True)).cuda()
                    elif 'avg' in self.cfg.TRAINER.CALIBRATE_TEXT:
                        self.domain_feat_avg[name] = self.text_features / self.text_features.norm(dim=-1, keepdim=True) - (self.domain_feat_avg['avg'] / self.domain_feat_avg['avg'].norm(dim=-1, keepdim=True)).cuda()
                    # import pdb; pdb.set_trace()

                else:
                    self.domain_feat_avg[name] = self.domain_feat_avg[name] - self.domain_feat_avg[self.main_domain]  #! 减去的是main domain的feature

                self.domtext_bias_features[name] = self.domain_feat_avg[name].cuda()

        if 'ensemble' in self.cfg.TRAINER.CALIBRATE_TEXT:
            if 'pig2' in self.cfg.TRAINER.CALIBRATE_TEXT:
                all_direct = []
                for name in list(keys):
                    all_direct.append(self.domtext_bias_features[name])
                all_direct = torch.stack(all_direct, dim=0).cuda()
                all_direct = all_direct.squeeze(dim=1)
                # import pdb; pdb.set_trace()
                # check all direct is [6, dim]
                self.text_bias_features = ada_lambda.mm(all_direct)
            else:
                self.text_bias_features = sum(self.domtext_bias_features.values()).cuda()
                    
  
@TRAINER_REGISTRY.register()
class ZeroshotCLIP_calibrate_v3_ensemble_summary(ZeroshotCLIP_calibrate_v3_ensemble):
    
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
                        count = 0
                        all_count = 0
                        for batch_idx, batch in enumerate(tqdm(loader)):
                            input, label = self.parse_batch_test(batch)
                            domlabel = batch['domlabel']    #! domain pred: cluster name
                            count += sum(label ==65)
                            all_count += label.shape[0]
            
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

            return 0   # 仅涉及测试阶段，无需保存current result



@TRAINER_REGISTRY.register()
class ZeroshotCLIP_calibrate_v3_multi_summary(ZeroshotCLIP_calibrate_v3_multi):
    templates = DOMAINNET_TEMPLATES
    
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
            alpha_list = np.arange(0.0, 1.1, 0.1)
            beta_list = np.arange(0.0, 1.4, 0.1)
            for alpha in alpha_list:
                self.alpha = alpha
                for beta in beta_list:
                    self.beta = beta

                    for domain, loader in data_loader.items():
                        print(f'Test Accuracy on {domain}' )
                        cur_evaluator.reset()
                        count = 0
                        all_count = 0
                        for batch_idx, batch in enumerate(tqdm(loader)):
                            input, label = self.parse_batch_test(batch)
                            domlabel = batch['domlabel']    #! domain pred: cluster name
                            count += sum(label ==65)
                            all_count += label.shape[0]
            
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

            return 0   # 仅涉及测试阶段，无需保存current result



