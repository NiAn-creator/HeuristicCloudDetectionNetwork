from tqdm import tqdm
import network
import os
import random
import argparse
import numpy as np
from torch.utils import data
from datasets import wscd_trainval,wscd_test,wscd_test_wdcd,wscd_test_landsat
from metrics import StreamSegMetrics
import torch
import torch.nn as nn
import time


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--dataset", type=str, default='gf1', help='Name of dataset')
    parser.add_argument("--data_root", type=str, default='./GF1_datasets/',
                        help="path to Dataset")

    parser.add_argument("--gpu_id", type=str, default='0',help="GPU ID")
    parser.add_argument("--batch_size", type=int, default=4,help='batch size (default: 4)')
    parser.add_argument("--num_classes", type=int, default=2,help="num classes (default: None)")

    # Model Options
    parser.add_argument("--model", type=str, default='mResNet34_RAPL_DBRM',
                        choices=['RAPL_DBRM_GF1','RAPL_DBRM512_GF1','RAPL_BRM_GF1','RAPL_BRM_GF1'], help='model name')

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=True)
    parser.add_argument("--ckpt", default='./checkpoints/best_epoch.pth',
                        help="restore from checkpoint")

    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"predict_path\"")
    parser.add_argument("--predict_path", default='./weakly_spuervisied_CD',
                        help="save prediction results")

    parser.add_argument("--random_seed", type=int, default=1,help="random seed (default: 1)")
    return parser

def get_dataset(opts):
    if opts.dataset == 'gf1':
        val_dst = wscd_test_gf1(root=opts.data_root, image_set='test')

    if opts.dataset == 'WDCD':
        val_dst = wscd_test_wdcd(root=opts.data_root, image_set='test')

    return  val_dst



def validate(opts, model, loader, metrics, device, threshold):
    metrics.reset()

    index = 0

    with torch.no_grad():
        for i,(images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs, edges = model(images)
            outputs = torch.squeeze(outputs).cpu().numpy()
            targets = labels.cpu().numpy()

            b, h, w = targets.shape[0], targets.shape[1], targets.shape[2]
            preds = np.zeros((b, h, w), dtype=int)
            preds[outputs >= threshold] = 1

            metrics.update(targets, preds)

            # outputs, edges, cloud_cam, d0_pred= model(images)
            # outputs = torch.squeeze(outputs).cpu().numpy()
            # edges = torch.squeeze(edges).cpu().numpy()
            # cloud_cam = torch.squeeze(cloud_cam).cpu().numpy()
            # d0_pred = torch.squeeze(d0_pred).cpu().numpy()
            #
            # tiff = images.cpu().numpy()
            # targets = labels.cpu().numpy()
            #
            # b, h, w = targets.shape[0], targets.shape[1], targets.shape[2]
            # preds = np.zeros((b, h, w), dtype=int)
            # preds[outputs >= threshold] = 1
            #
            # metrics.update(targets, preds)

            if opts.save_val_results:
                sample_fname = loader.sampler.data_source.masks[:]
                os.makedirs(opts.predict_path, exist_ok=True)
                os.makedirs(opts.predict_path + 'oup/', exist_ok=True)
                # os.makedirs(opts.predict_path + 'edg/', exist_ok=True)
                # os.makedirs(opts.predict_path + 'cam/', exist_ok=True)
                # os.makedirs(opts.predict_path + 'd0/', exist_ok=True)
                # print('Save position is %s\n' % (opts.predict_path))

                for batch in range(images.shape[0]):
                    content = sample_fname[index].replace('.npy', "").split("/")
                    name = content[-1]

                    # print('%d ---------%s---------' % (index, name))
                    np.save(opts.predict_path + 'oup/' + name + '_0oup.npy', preds[batch,  :, :])
                    # np.save(opts.predict_path + 'edg/' + name + '_1edg.npy', edges[batch, :, :])
                    # np.save(opts.predict_path + 'cam/' + name + '_2cam.npy', cloud_cam[batch, :, :])
                    # np.save(opts.predict_path + 'd0/' + name + '_3d0.npy', d0_pred[batch, :, :])
                    index = index + 1

        score = metrics.get_results()
    return score,threshold


if __name__ == '__main__':
    opts = get_argparser().parse_args()

    # select the GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s,  CUDA_VISIBLE_DEVICES: %s\n" % (device, opts.gpu_id))

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # train_dst, val_dst = get_dataset(opts)
    val_dst = get_dataset(opts)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size, shuffle=False, num_workers=8, drop_last=True,
                                 pin_memory=False)

    print("Dataset: %s, val set: %d" % (opts.dataset, len(val_dst)))

    # Set up model
    model_map = {
        'mResNet50_RAPL_BRM': network.mResNet50_RAPL_BRM_GF1,
        'mResNet50_RAPL_DBRM': network.mResNet50_RAPL_DBRM_GF1,
        'mResNet34_RAPL_noBRM':network.mResNet34_RAPL_noBRM_GF1,
        'mResNet34_RAPL_BRM': network.mResNet34_RAPL_BRM_GF1,
        'mResNet34_RAPL_DBRM': network.mResNet34_RAPL_DBRM_GF1,
        'VGG16_RAPL_BRM':network.VGG16_RAPL_BRM_GF1,
        'VGG16_RAPL_DBRM': network.VGG16_RAPL_DBRM_GF1,
    }

    print('Model = %s, num_classes=%d' % (opts.model, opts.num_classes))
    model = model_map[opts.model](num_classes=opts.num_classes)

    # Restore
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Model restored from %s" % opts.ckpt)
    else:
        print("Error: Can not load best checkpoints.")

    if opts.test_only:
        model.eval()

        for thres in [0.4]:
            print('************************************************************************')
            print(thres)
            time_before_val = time.time()

            val_score,threshold = validate(opts=opts, model=model, loader=val_loader,
                                           metrics=metrics, device=device,threshold=thres)

            time_after_val = time.time()
            print('Time_val = %f' % (time_after_val - time_before_val))
            print('Threshold = %f' % (threshold))
            print(metrics.to_str(val_score))