from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from einops import rearrange, repeat

import math
import torch
import torch.nn as nn
from utils.visualizer import Visualizer
from network.fcn8s_resnet import FCN8s
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from datasets import av_corrected


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

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50','deeplabv3plus_resnet50_DM','deeplabv3plus_resnet50_drop','deeplabv3plus_resnet50_DMv3v2',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101','deeplabv3plus_resnet101_DM','deeplabv3plus_resnet50_DMv4','deeplabv3plus_resnet50_DMv3v4',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet','FCN_resnet50','deeplabv3plus_resnet50_DMv3v3','deeplabv3plus_resnet50_DMv3v5',
                                 'deeplabv3plus_spectral50','deeplabv3plus_resnet50_DMv2'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=50e3,
                        help="epoch number (default: 50k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
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
    parser.add_argument("--ckptpath", type=str, default='checkpoints', help="folder where to save the ckt (default: checkpoints)")

    return parser

#nb_proto=88
nb_proto=22
def get_dataset(opts):
    """ Dataset And Augmentation
    """
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
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        if opts.model=='deeplabv3plus_spectral50':
            train_transform = et.ExtCompose([
                #et.ExtResize( 512 ),
                et.ExtRandomCrop( size=(512,1024) , pad_if_needed=True),
                et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
                et.ExtRandomHorizontalFlip(),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])

            val_transform = et.ExtCompose([
                #et.ExtResize( 512 ),
                et.ExtScale(0.5),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
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

    if opts.dataset == 'av':
        if opts.model=='deeplabv3plus_spectral50':
            train_transform = et.ExtCompose([
                # et.ExtResize( 512 ),
                et.ExtRandomScale((0.5, 2.0)),
                et.ExtRandomCrop( size=(512,1024) , pad_if_needed=True),
                et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                et.ExtRandomHorizontalFlip(),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
            val_transform = et.ExtCompose([
                # et.ExtResize( 512 ),
                et.ExtScale(0.5),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
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
        val_dst = av_corrected.dataset(root_dataset=opts.data_root, root_odgt=opts.odgt_root,
                             split = 'val', transform=val_transform)

    return train_dst, val_dst

def centered_cov_torch(x):
    n = x.shape[0]
    res =1 /n *x.t().mm(x)
    print('00000000000000000000000000000000',x.max(),x.min(),torch.isinf(x).sum(),torch.isinf(res).sum(),'n',n)
    #res =1 /n * res
    #res = x.t().mm(x)
    return res

def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()
max_iter=5
def gmm_fit_v1(model,loader,device):
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            name_img='bad'

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            embeddings, conf = model.module.compute_features(images)
            _, proto_labels  = torch.max(embeddings,dim=1)
            b,c,h,w=embeddings.size()

            embeddings_tmp = embeddings[:, 0, :, :]
            embeddings_new = embeddings_tmp[labels != 255]
            embeddings_new = torch.unsqueeze(embeddings_new, 1)
            for idx in range(1,c):
                embeddings_tmp=embeddings[:,idx,:,:]
                embeddings_tmp=embeddings_tmp[labels != 255]
                embeddings_tmp = torch.unsqueeze(embeddings_tmp, 1)
                embeddings_new =torch.cat((embeddings_new, embeddings_tmp), 1)
            print('embeddings_new',embeddings_new.size())
            permuation=np.random.permutation(len(embeddings_new))
            embeddings_sample=embeddings_new[permuation[0:50000]]
            if i==0:
                classwise_mean_features = torch.mean(embeddings_sample, dim=0)
                classwise_cov_features= cov(embeddings_sample)
                incr=1

            else:
                incr+=1
                classwise_mean_features+= torch.mean(embeddings_sample, dim=0)
                classwise_cov_features+= cov(embeddings_sample)#centered_cov_torch(embeddings_new- classwise_mean_features)


            if i >max_iter: break



        classwise_cov_features_norm=classwise_cov_features/incr
        #classwise_mean_features_norm=torch.stack([classwise_mean_features[c]/classwise_incr[c].float()  for c in range(nb_proto)])
        classwise_mean_features_norm=classwise_mean_features/incr

        #gmm = torch.distributions.MultivariateNormal(loc=classwise_mean_features_norm.cpu().float(), covariance_matrix=classwise_cov_features_norm.cpu().float(), )

        classwise_mean_features_norm=classwise_mean_features_norm.cpu().numpy().astype(np.float32)
        classwise_cov_features_norm=classwise_cov_features_norm.cpu().numpy().astype(np.float32)
        classwise_cov_features_norm+=0.1*np.eye(c, dtype=int)
        classwise_cov_features_normINV=np.linalg.inv(classwise_cov_features_norm)
        denom = np.linalg.det((2*math.pi*0.1*np.eye(c, dtype=int)))**0.5 #np.linalg.det((2*math.pi*classwise_cov_features_norm))
        gmm={"mean":classwise_mean_features_norm,"cov":classwise_cov_features_norm,"cov_inv":classwise_cov_features_normINV,"denom":denom}
        del embeddings_new
        del embeddings_sample
        del embeddings_tmp
        del embeddings
        del images



    return gmm

def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0
    mode_dico= []
    sum_pixels=0
    for i in range(nb_proto): mode_dico.append(0)
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            embeddings_1batch,conf = model.module.compute_features(images)
            b, c, h, w=images.size()
            sum_pixels+=b * h * w

            _, pred_proto = embeddings_1batch.detach().max(1)
            conf =1-torch.squeeze(torch.sigmoid(conf))
            if i==0:
                name_img0='checking_lossloss_BCE_prop5imask_.jpg'
                conf0=conf[0]
                conf0=conf0-conf0.min()
                conf0=conf0/conf0.max()
                img_conf=((conf0* 255).detach().cpu().numpy()).astype(np.uint8)
                Image.fromarray(img_conf).save('results/new_VOS/'+ name_img0)
            for i in range(nb_proto): mode_dico[i] += (pred_proto == i).float().sum().item()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if i % 50 == 0:
                if opts.save_val_results:
                    for i in range(len(images)):
                        image = images[i].detach().cpu().numpy()
                        target = targets[i]
                        pred = preds[i]

                        image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                        target = loader.dataset.decode_target(target).astype(np.uint8)
                        pred = loader.dataset.decode_target(pred).astype(np.uint8)

                        if not os.path.isdir(f'results/{opts.model}'):
                            os.mkdir(f'results/{opts.model}')
                        Image.fromarray(image).save(f'results/{opts.model}/%d_image.png' % img_id)
                        Image.fromarray(target).save(f'results/{opts.model}/%d_target.png' % img_id)
                        Image.fromarray(pred).save(f'results/{opts.model}/%d_pred.png' % img_id)

                        fig = plt.figure()
                        plt.imshow(image)
                        plt.axis('off')
                        plt.imshow(pred, alpha=0.7)
                        ax = plt.gca()
                        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                        plt.savefig(f'results/{opts.model}/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                        plt.close()
                        img_id += 1

        score = metrics.get_results()
        print('score collapsing prototypes =',np.amax(np.array(mode_dico))/sum_pixels )
    return score, ret_samples

def normpdf(x, gmm):
    n = x.shape[0]
    var = 0.1#float(sd)**2

    out=[]
    for i in range(n):
        num =np.exp(-0.5*np.matmul(np.matmul((x[i]-gmm['mean']), gmm['cov_inv']),np.transpose((x[i]-gmm['mean']))))/gmm['denom']
        #print(np.shape(num),num)
        out.append(num)
    out=np.array(out)

    return out

def generate_cutout_mask(img_size, seed = None):
    np.random.seed(seed)

    cutout_area = img_size[0] * img_size[1] / 8

    w = np.random.randint(img_size[1] / 2, img_size[1])
    h = np.amin((np.round(cutout_area / w),img_size[0]))

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = np.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.astype(float)


def main():
    opts = get_argparser().parse_args()
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

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model
    if 'deeplabv3' in opts.model:
        model_map = {
            'deeplabv3_resnet50': network.deeplabv3_resnet50,
            'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
            'deeplabv3plus_resnet50_DM': network.deeplabv3plus_resnet50_DM,
            'deeplabv3plus_resnet50_drop': network.deeplabv3plus_resnet50_drop,
            'deeplabv3plus_resnet50_DMv2': network.deeplabv3plus_resnet50_DM_v2,
            'deeplabv3plus_resnet50_DMv3v2': network.deeplabv3plus_resnet50_DM_v3v2,
            'deeplabv3plus_resnet50_DMv3v3': network.deeplabv3plus_resnet50_DM_v3v3,
            'deeplabv3plus_resnet50_DMv3v4': network.deeplabv3plus_resnet50_DM_v3v4,
            'deeplabv3plus_resnet50_DMv3v5': network.deeplabv3plus_resnet50_DM_v3v5,
            'deeplabv3plus_resnet50_DMv4': network.deeplabv3plus_resnet50_DM_v4,
            'deeplabv3plus_resnet101_DM': network.deeplabv3plus_resnet101_DM,
            'deeplabv3_resnet101': network.deeplabv3_resnet101,
            'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
            'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
            'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet,
            'deeplabv3plus_spectral50':network.deeplabv3plus_spetralresnet50
        }
        if opts.model=='deeplabv3plus_spectral50': model = model_map[opts.model](inputsize1=512,inputsize2=1024,num_classes=opts.num_classes, output_stride=opts.output_stride)
        else: model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
        if opts.separable_conv and 'plus' in opts.model:
            network.convert_to_separable_conv(model.classifier)
        utils.set_bn_momentum(model.backbone, momentum=0.01)
    elif 'FCN_resnet50' == opts.model:
        model = FCN8s( opts.crop_size, spectral_normalization=True, pretrained=False, n_class=opts.num_classes)


    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)
    
    # Set up optimizer
    if 'deeplabv3' in opts.model:
        optimizer = torch.optim.SGD(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr},
        ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    elif 'FCN_resnet50' == opts.model:
        optimizer = torch.optim.SGD(params= model.parameters(), lr= opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

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
        criterion_new = nn.CrossEntropyLoss(ignore_index=255,reduction='none')
        criterionMSE = torch.nn.MSELoss()
        criterionBCE = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).cuda()) #torch.tensor([50.0]).cuda())

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

    utils.mkdir(opts.ckptpath)
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        #model.to(device).to_fp16()
        print('we have loaded ',opts.ckpt)
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
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model.to(device)

    #==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    scaler = torch.cuda.amp.GradScaler()
    torch.cuda.empty_cache()
    Softmax = torch.nn.Softmax2d()
    with torch.cuda.amp.autocast():
        model.eval()
        gmm =gmm_fit_v1(model,train_loader,device)

    print('gmm',gmm)
    for i in range(10):
        x = np.random.multivariate_normal(gmm['mean'], gmm['cov'], (500000))
        proba=normpdf(x, gmm)
        x_sample0=x[proba<1.0e-30]
        if i==0: x_sample=x_sample0
        else:x_sample=np.concatenate((x_sample, x_sample0), axis=0)
        print('proba',proba,np.max(proba),np.min(proba),'///',np.shape(x_sample))

    while cur_epochs<80: #while True: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            permutation = np.random.permutation(len(x_sample))
            x_sample_gpu = torch.from_numpy(x_sample[permutation])


            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)


                loss_CE = criterion_new(outputs, labels).detach()
                loss_CEdetached = loss_CE
                loss_CEdetached = loss_CEdetached - loss_CEdetached.min()
                loss_CEdetached = loss_CEdetached/loss_CEdetached.max()
                loss_CEdetached[labels == 255] = 1
                #embeddings_1batch,conf = model.module.compute_features(images)
                embeddings_1batch, embeddings_1batch ,conf = model.module.compute_features1(images)
                img_size = embeddings_1batch.shape[2:4]
                x_sample_gpu = x_sample_gpu[opts.batch_size*img_size[0]*img_size[1]]
                #x_sample_gpu = x_sample_gpu.to(device, dtype=torch.float16)
                for image_i in range(opts.batch_size):
                    if image_i == 0:
                        MixMask = torch.from_numpy(generate_cutout_mask(img_size)).unsqueeze(0).to(device,                                                                         dtype=torch.float16)
                    else:
                        Mask = torch.from_numpy(generate_cutout_mask(img_size)).unsqueeze(0).to(device,dtype=torch.float16)
                        MixMask = torch.cat((MixMask, Mask))


                '''data = torch.cat(
                    [(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in
                     range(data.shape[0])])'''
                print(MixMask.size(),'//////',embeddings_1batch.size(),'//////',x_sample_gpu.size())
                img_conf=((torch.squeeze(Mask)* 255).detach().cpu().numpy()).astype(np.uint8)
                name_img0='mask_.jpg'
                Image.fromarray(img_conf).save('results/new_VOS/'+ name_img0)
                #print(conf)
                embeddings_proba=Softmax(embeddings_1batch)
                embeddings_entropy =torch.sum(embeddings_proba*torch.log(embeddings_proba),dim=1)
                #embeddings_1batch = torch.mean(embeddings_1batch,dim=1)
                loss_entropy=torch.mean(embeddings_entropy)
                loss_CEdetached=torch.unsqueeze(loss_CEdetached, 1)
                #loss_MSE= criterionMSE(conf,loss_CEdetached)
                #print(conf,loss_CEdetached ,'conf',conf.size(),'loss_CEdetached',loss_CEdetached.size())

                #loss_BCE= criterionBCE(conf,loss_CEdetached)
                #print(loss_BCE)
                #conf = torch.mean(embeddings_1batch,dim=1)
                #loss_kmeans=torch.mean(torch.abs(loss_CE.detach()-conf))#-0.1*loss_proto

                loss_proto = model.module.loss_kmeans()
                
                loss = criterion(outputs, labels)+0.1*loss_entropy-0.1*loss_proto #+0.1*loss_BCE
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #loss.backward()
            #optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss/10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt(opts.ckptpath+'/latest_DM_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt(opts.ckptpath+'/best_DM_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset,opts.output_stride))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()

            if cur_itrs >=  opts.total_itrs:
                return


if __name__ == '__main__':
    main()

