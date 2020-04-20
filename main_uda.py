from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from network.modeling import *
from network.discriminator import VGG, Discriminator
from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data/GTA',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='GTA',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30000,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--dis_lr", type=float, default=0.001,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=16,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512)
    
    parser.add_argument("--ckpt", default='', type=str,
                        help="restore from checkpoint")
    parser.add_argument("--ckpt_discriminator", default='', type=str,
                        help="restore from discriminator checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)
    parser.add_argument("--discriminator_loss_threshold", type=float, default=0.1,
                        help='discriminator_loss_threshold (default: 0.1)')
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
    parser.add_argument("--enable_vis", action='store_true', default=True,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='8097',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """

    if opts.dataset == 'GTA':
        # set transform parameters of pre-process
        train_transform = et.ExtCompose([
            #et.ExtResize( 768 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            #et.ExtResize( 768 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        # read data and transform
        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
        # train_city = Cityscapes(root=opts.data_root, use_cityscape = True,
        #                        split='train', transform=train_transform)
        # val_city = Cityscapes(root=opts.data_root, use_cityscape = True,
        #                        split='val', transform=train_transform)
    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []

    # set save validate results parameters
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    # validate
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))
            # save validate results
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

                    # fig = plt.figure()
                    # plt.imshow(image)
                    # plt.axis('off')
                    # plt.imshow(pred, alpha=0.7)
                    # ax = plt.gca()
                    # ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    # ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    # plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    # plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    opts.num_classes = 19

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))
    # set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=64)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=8)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model
    model_map = {
#        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.modeling.deeplabv3plus_mobilenet
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
 
    discriminator = Discriminator()


    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.Adam(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr)
    # if opts.lr_policy=='poly':
    #     scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    # elif opts.lr_policy=='step':
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    opt_dis = torch.optim.Adam(discriminator.parameters(), lr=opts.dis_lr)


    # Set up criterion
    #criterion = utils.get_loss(opts.loss_type)
    #loss set
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    #save path
    def save_ckpt(path, path_d):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            # "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": discriminator.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            # "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path_d)
        print("discriminator saved as %s" % path_d)

    
    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state'])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            # scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)
    

    if opts.ckpt_discriminator is not None and os.path.isfile(opts.ckpt_discriminator):
        checkpoint = torch.load(opts.ckpt_discriminator, map_location=torch.device('cpu'))
        discriminator.load_state_dict(checkpoint['model_state'])
        discriminator = nn.DataParallel(discriminator)
        discriminator.to(device)
        print("discriminator restored from %s" % opts.ckpt_discriminator)
        del checkpoint  # free memory
    else:
        discriminator = nn.DataParallel(discriminator)
        discriminator.to(device)        



    print(discriminator)
    print(model)



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

    interval_loss = 0.0
    interval_task_loss = 0.0
    interval_adverserial_loss = 0.0
    interval_Discriminator_loss = 0.0

    dis_label_s = np.zeros((opts.batch_size))
    dis_label_t = np.ones((opts.batch_size))

    dis_label_s = Variable(torch.from_numpy(np.array(dis_label_s)).long(),requires_grad=False).to(device)
    dis_label_t = Variable(torch.from_numpy(np.array(dis_label_t)).long(),requires_grad=False).to(device)

    gan_criterion = nn.CrossEntropyLoss().cuda()

    train_generator = False

    while True: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        discriminator.train()
        cur_epochs += 1
        for (images, labels, city_imgs) in train_loader:
            if images.shape[0] != opts.batch_size:
                continue
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            city_imgs = city_imgs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)


            optimizer.zero_grad()
            opt_dis.zero_grad()

            ## train discriminator

            if not train_generator or (cur_itrs) % 10 == 0:

                features_s = model(images,return_feature=True)
                f_s = Variable(features_s.data)
                score_s = discriminator(f_s.detach())
                discriminator_loss_s = gan_criterion(score_s, dis_label_s)

                features_t = model(city_imgs,return_feature=True)
                f_t = Variable(features_t.data, requires_grad=False)
                score_t = discriminator(f_t)
                discriminator_loss_t = gan_criterion(score_t, dis_label_t)

                discriminator_loss = (discriminator_loss_s + discriminator_loss_t)/2

                discriminator_loss.backward()

            # print(score_t.detach().cpu().numpy(), score_s.detach().cpu().numpy(),
            #     discriminator_loss_s.detach().cpu().numpy(),discriminator_loss_t.detach().cpu().numpy(), discriminator_loss.detach().cpu().numpy())

            opt_dis.step()


            # train generator

            if discriminator_loss < opts.discriminator_loss_threshold:
                train_generator = True
            if train_generator:
                outputs = model(images)
                task_loss = criterion(outputs, labels)
                score_s = discriminator(f_s)
                adverserial_loss = gan_criterion(score_s,dis_label_t)

                loss = task_loss + adverserial_loss

                loss.backward()
                optimizer.step()

                np_loss = loss.detach().cpu().numpy()
                np_task_loss = task_loss.detach().cpu().numpy()
                np_adverserial_loss = adverserial_loss.detach().cpu().numpy()
            
                interval_loss += np_loss
                interval_task_loss += np_task_loss
                interval_adverserial_loss += np_adverserial_loss


            np_Discriminator_loss = discriminator_loss.detach().cpu().numpy()
            interval_Discriminator_loss += np_Discriminator_loss
            
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_Discriminator_loss)

            if (cur_itrs) % 10 == 0:

                print("Epoch %d, Itrs %d/%d, loss: D=%f, G total= %f, task= %f, adverserial= %f" %
                      (cur_epochs, cur_itrs, opts.total_itrs,interval_Discriminator_loss/10, interval_loss/10,interval_task_loss/10,interval_adverserial_loss/10 ))
                
                interval_loss = 0.0
                interval_task_loss = 0.0
                interval_adverserial_loss = 0.0
                interval_Discriminator_loss = 0.0


            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest_uda_GTA_to_cityscape.pth','checkpoints/latest_uda_Discriminator.pth')
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_uda_GTA_to_cityscape.pth','checkpoints/best_uda_Discriminator.pth')

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
            # scheduler.step()  

            if cur_itrs >=  opts.total_itrs:
                return

        
if __name__ == '__main__':
    main()