from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import time

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from einops import rearrange, repeat

import torch
import torch.nn as nn
from utils.visualizer import Visualizer
import anom_utils
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from datasets import av_corrected

consistency_loss = 'CE'
consistency_weight = 1

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--odgt_root", type=str, default='./datasets/data',
                        help="path to odgt file")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes', 'av'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")
    parser.add_argument("--phase", type=str, default='val',
                        choices=['val','test_normal',  'test_OOD','test_level1','test_level2'], help='phase choice')
    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50','deeplabv3plus_resnet50_DM','deeplabv3plus_resnet50_drop','deeplabv3plus_resnet50_DMv3v2',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101','deeplabv3plus_resnet101_DM','deeplabv3plus_resnet50_DMv4',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet','FCN_resnet50','deeplabv3plus_resnet50_DMv3v3',
                                 'deeplabv3plus_spectral50','deeplabv3plus_resnet50_DMv2'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=2,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    # Mixing strategy
    parser.add_argument("--mix", type=str, default='watershedmix',
                        choices=['watershedmix',  'cut'], help='mixing strategy name choose between cutmix -> cut and watershedmix')

    return parser
nb_proto=22
class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, confidences, predictions, labels):

        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=confidences.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

ECE = _ECELoss()

def eval_ood_measure(conf, seg_label,name, mask=None):
    conf= conf.cpu().numpy()
    seg_label= seg_label.cpu().numpy()
    out_labels = [16,17,18]
    if mask is not None:
        seg_label = seg_label[mask]

    out_label = seg_label == out_labels[0]
    for label in out_labels:
        out_label = np.logical_or(out_label, seg_label == label)

    in_scores = - conf[np.logical_not(out_label)]
    out_scores  = - conf[out_label]

    if (len(out_scores) != 0) and (len(in_scores) != 0):
        auroc, aupr, fpr = anom_utils.get_and_print_results(out_scores, in_scores)
        return auroc, aupr, fpr
    else:
        print("This image does not contain any OOD pixels or is only OOD.")
        print(name)
        return None

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    mix_mask = opts.mix
    watershed=False
    if mix_mask == 'watershedmix': watershed =True
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            #et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst_labelled = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        train_dst_unlabelled = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)
    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        val_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
        '''val_dst = Cityscapes_Rain(root=opts.data_root,
                             split='val', transform=val_transform)'''

    if opts.dataset == 'av':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = av_corrected.dataset(root_dataset=opts.data_root, root_odgt=opts.odgt_root,
                               split = 'train', transform=train_transform)
        if opts.phase=='val':
            ['val','test_normal',  'test_OOD','test_level1','test_level2']
            val_dst = av_corrected.dataset(root_dataset=opts.data_root, root_odgt=opts.odgt_root,
                             split = 'val', transform=val_transform)
        elif opts.phase=='test_normal':
            val_dst = av_corrected.dataset(root_dataset=opts.data_root, root_odgt=opts.odgt_root,
                             split = 'test_normal', transform=val_transform)
        elif opts.phase=='test_OOD':
            val_dst = av_corrected.dataset(root_dataset=opts.data_root, root_odgt=opts.odgt_root,
                             split = 'test_OOD', transform=val_transform)
        elif opts.phase=='test_level1':
            val_dst = av_corrected.dataset(root_dataset=opts.data_root, root_odgt=opts.odgt_root,
                             split = 'test_level1', transform=val_transform)
        elif opts.phase=='test_level2':
            val_dst = av_corrected.dataset(root_dataset=opts.data_root, root_odgt=opts.odgt_root,
                             split = 'test_level2', transform=val_transform)
    return train_dst, val_dst


def centered_cov_torch(x):
    n = x.shape[0]
    res = 1 /n * x.t().mm(x)
    #res = x.t().mm(x)
    return res

def moment1_torch(x):
    n = x.shape[0]
    res = x.t().mm(x)
    return res


DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-308, 0, 1)]
max_iter =20
def oneclass_fit_v1(model,loader,device):
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            name_img='bad'

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            embeddings, conf = model.module.compute_features(images)
            _, proto_labels  = torch.max(embeddings,dim=1)
            b,c,h,w=embeddings.size()
            embeddings_tmp = embeddings[:, c, :, :]
            embeddings_new = embeddings_tmp[labels != 255]
            embeddings_new = torch.unsqueeze(embeddings_new, 1)
            for i in range(1,c):
                embeddings_tmp=embeddings[:,c,:,:]
                embeddings_tmp=embeddings_tmp[labels != 255]
                embeddings_tmp = torch.unsqueeze(embeddings_tmp, 1)
                embeddings_new =torch.cat((embeddings_new, embeddings_tmp), 0)
        embeddings = embeddings_new
        print('embeddings_new',embeddings.size())



    return gmm, jitter_eps


def validate(opts, model, loader,loader_train, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    LogSoftmax = nn.LogSoftmax(dim=1) #torch.nn.Softmax2d()
    Softmax = torch.nn.Softmax2d()
    ret_samples = []
    ece=[]
    NLL=[]
    auroc_list=[]
    aupr_list=[]
    fpr_list=[]
    model.eval()
    maximum_all_data=[]
    criterion_CE = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    criterion_NLL = nn.NLLLoss(ignore_index=255, reduction='mean')
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    gaussians_model_DDU, jitter_eps =gmm_fit_v1(model,loader_train,device)

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            name_img='bad'

            #print('   name_img',name_img[0].split('/')[-1]  )
            name_img0=name_img[0].split('/')[-1]
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            outputs_feature,conf = model.module.compute_features(images)

            outputs_feature_tmp = rearrange(outputs_feature, 'b h n d -> b n d h')

            #log_probs = gaussians_model.log_prob(outputs_feature[:,:, :, :])
            #out0=torch.reshape(outputs_feature_tmp[0] ,(2048*1024,22))
            print(outputs_feature_tmp.size())
            out0=torch.reshape(outputs_feature_tmp[0] ,(2048*1024,22))
            #out1=torch.reshape(outputs_feature[1] ,(2048*1024,22))
            #print(out0)
            #print('out0',out0.size())
            log_probs0 = gaussians_model_DDU.log_prob(out0[:, None, :].cpu())
            #conf_0, pred0 = log_probs0.max(1)
            #conf_0 =torch.mean(log_probs0,dim=1)
            
            #conf_0 = torch.reshape(conf_0, (1024, 2048)).cuda()
            outputs_feature_tmp = torch.reshape(log_probs0, (1,1024, 2048,22)).cuda()
            outputs_feature_tmp = rearrange(outputs_feature_tmp, 'b h w c -> b c h w')
            outputs_feature_tmp = torch.sigmoid(outputs_feature_tmp)#
            conf_0=model.module.classifier.conv1x1(outputs_feature_tmp)
            #conf_0 = (1 - torch.squeeze(torch.sigmoid(conf_0)))#
            conf_0=torch.squeeze(conf_0)
            conf_0 =conf_0 - conf_0.min()
            conf_0 =conf_0/conf_0.max()
            print('111111111 unique',torch.unique(conf_0))
            #print(outputs_feature.size(),outputs_feature.max(),outputs_feature.min())
            conf0, preds = outputs.detach().max(dim=1)
            preds=preds.cpu().numpy()
            #preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            outputsproba=Softmax(outputs)
            outputslogproba=LogSoftmax(outputs).type(torch.float64)

            nll_out=criterion_NLL(outputslogproba, labels)
            #nll_out=criterion_CE(outputs.type(torch.float64), labels)
            #print(nll_out,criterion_CE(outputs.type(torch.float64), labels))
            NLL.append(nll_out.cpu().item())
            _, preds_val  = torch.max(outputsproba,dim=1)
            sorted, _= torch.sort(outputsproba,dim=1,descending=True)
            #conf = torch.mean(outputs_feature,dim=1)
            _, preds_proto  = torch.max(outputs_feature,dim=1)

            conf = (1 - torch.squeeze(torch.sigmoid(conf)))#*conf0

            conf = conf/conf.max()
            print(name_img0)
            print(preds_proto.size())
            name_img0=name_img0+str(i)+'.jpg'
            preds_proto0=preds_proto/preds_proto.max()
            img_all =torch.cat((preds_proto0[0], conf[0],conf_0), dim=1)
            #img_conf=((preds_proto0[0]* 255).detach().cpu().numpy()).astype(np.uint8)
            img_conf=((img_all* 255).detach().cpu().numpy()).astype(np.uint8)
            print('name_img0',name_img0,np.shape(img_conf))#,img_conf)
            Image.fromarray(img_conf).save('results/new_BCE/'+ name_img0)
            print('np.unique(img_conf)',np.unique(img_conf))
            #conf,_=torch.max(outputs_feature,dim=1)
            #conf = 3*conf
            #print(conf.min(),conf.max())

            #conf = conf/1.1#2.7
            #print(conf.min(),conf.max())

            maximum_all_data.append(conf.max().item())
            '''OO=outputs_feature[0,:,0:10,0]
            OO=torch.transpose(OO, 0, 1)'''

            mask = None
            auroc, aupr, fpr =  0,0,0
            if  (opts.phase=='test_OOD') or (opts.phase=='test_level1') or (opts.phase=='test_level2'):
                try:
                    auroc, aupr, fpr = eval_ood_measure(conf, labels,name_img, mask=mask)
                except:
                    print('pb with image name_img =',name_img)
            #print(conf.size(),preds_val.size())
            #print(auroc, aupr, fpr)
            conf=sorted[:,0]/(sorted[:,1]+0.1)#0.05)
            conf=conf/10#20
            #conf0
            #conf = 1-conf0
            #conf = conf/conf.max()
            ###print(name_img0)
            '''img_conf=((conf[0]* 255).detach().cpu().numpy()).astype(np.uint8)
            print('name_img0',name_img0,np.shape(img_conf),img_conf)
            Image.fromarray(img_conf).save('results/new_BCE/'+ name_img0)
            print('np.unique(img_conf)',np.unique(img_conf))'''

            #print(conf.min(),conf.max())
            #conf=conf0


            ece_out =  ECE.forward(conf, preds_val,labels)
            #ece_out = ECE.forward(conf.squeeze(), preds_val.squeeze(),labels)
            ece.append(ece_out.cpu().item())
            auroc_list.append(auroc)
            aupr_list.append(aupr)
            fpr_list.append(fpr)


            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)



                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    print('np.max(maximum_all_data)',np.max(maximum_all_data),np.min(maximum_all_data))
    return score, ret_samples,ece,NLL, auroc_list, aupr_list, fpr_list

def create_ema_model(model,modelname,num_classes,output_stride,gpus):
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3plus_resnet50_DM': network.deeplabv3plus_resnet50_DM,
        'deeplabv3plus_resnet50_drop': network.deeplabv3plus_resnet50_drop,
        'deeplabv3plus_resnet50_DMv2': network.deeplabv3plus_resnet50_DM_v2,
        'deeplabv3plus_resnet50_DMv3v2': network.deeplabv3plus_resnet50_DM_v3v2,
        'deeplabv3plus_resnet50_DMv3v3': network.deeplabv3plus_resnet50_DM_v3v3,
        'deeplabv3plus_resnet50_DMv4': network.deeplabv3plus_resnet50_DM_v4,
        'deeplabv3plus_resnet101_DM': network.deeplabv3plus_resnet101_DM,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet,
        'deeplabv3plus_spectral50': network.deeplabv3plus_spetralresnet50
    }

    ema_model = model_map[modelname](num_classes=num_classes, output_stride=output_stride)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    if len(gpus)>1:
        ema_model = torch.nn.DataParallel(ema_model)
    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher, iteration,gpus):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    if len(gpus)>1:
        for ema_param, param in zip(ema_model.module.parameters(), model.module.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    else:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def mix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        if mask.shape[0] == data.shape[0]:
            data = torch.cat([(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])])
        elif mask.shape[0] == data.shape[0] / 2:
            data = torch.cat((torch.cat([(mask[i] * data[2 * i] + (1 - mask[i]) * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))]),
                              torch.cat([((1 - mask[i]) * data[2 * i] + mask[i] * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))])))
    if not (target is None):
        target = torch.cat([(mask[i] * target[i] + (1 - mask[i]) * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])
    return data, target

def generate_cutout_mask(img_size, seed = None):
    np.random.seed(seed)

    cutout_area = img_size[0] * img_size[1] / 2

    w = np.random.randint(img_size[1] / 2, img_size[1])
    h = np.amin((np.round(cutout_area / w),img_size[0]))

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = np.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.astype(float)

class CrossEntropyLoss2dPixelWiseWeighted(nn.Module):
    def __init__(self, weight=None, ignore_index=250, reduction='none'):
        super(CrossEntropyLoss2dPixelWiseWeighted, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target, pixelWiseWeight):
        loss = self.CE(output, target)
        loss = torch.mean(loss * pixelWiseWeight)
        return loss

class MSELoss2d(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean', ignore_index=255):
        super(MSELoss2d, self).__init__()
        self.MSE = nn.MSELoss(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, output, target):
        loss = self.MSE(torch.softmax(output, dim=1), target)
        return loss

def main():

    opts = get_argparser().parse_args()
    mix_mask = opts.mix
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'av':
        opts.num_classes = 19

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset=='voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst_labelled, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst_labelled, batch_size=opts.batch_size, shuffle=True, num_workers=2)

    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2, drop_last=True)
    #interp = nn.Upsample(size=(opts.crop_size, opts.crop_size), mode='bilinear', align_corners=True)

    if consistency_loss == 'CE':
         unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    elif consistency_loss == 'MSE':
        unlabeled_loss =  MSELoss2d().cuda()

    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst_labelled), len(val_dst)))

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3plus_resnet50_DM': network.deeplabv3plus_resnet50_DM,
        'deeplabv3plus_resnet50_drop': network.deeplabv3plus_resnet50_drop,
        'deeplabv3plus_resnet50_DMv2': network.deeplabv3plus_resnet50_DM_v2,
        'deeplabv3plus_resnet50_DMv3v2': network.deeplabv3plus_resnet50_DM_v3v2,
        'deeplabv3plus_resnet50_DMv3v3': network.deeplabv3plus_resnet50_DM_v3v3,
        'deeplabv3plus_resnet50_DMv4': network.deeplabv3plus_resnet50_DM_v4,
        'deeplabv3plus_resnet101_DM': network.deeplabv3plus_resnet101_DM,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet,
        'deeplabv3plus_spectral50': network.deeplabv3plus_spetralresnet50
    }


    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)



    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    #criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])

        model = nn.DataParallel(model)
        model.to(device)

        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)


    #==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images


    model.eval()
    val_score, ret_samples,ece,NLL, auroc_list, aupr_list, fpr_list  = validate(
        opts=opts, model=model, loader=val_loader,loader_train=train_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
    print(metrics.to_str(val_score))
    print('--------------------------------------------------------------------')
    print('ECE!!!!! mean ECE = ', np.mean(np.asarray(ece)))
    print('NLL!!!!! mean NLL = ', np.mean(np.asarray(NLL)))
    print('auroc!!!!! mean AUROC = ', np.mean(np.asarray(auroc_list)))
    print('aupr !!!!! mean AUPR = ', np.mean(np.asarray(aupr_list)))
    print('fpr!!!!! mean FPR = ', np.mean(np.asarray(fpr_list)))
    return





if __name__ == '__main__':
    main()
