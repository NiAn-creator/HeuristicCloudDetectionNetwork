import os
import argparse
import numpy as np
from metrics import StreamSegMetrics
import time

def get_argparser():
    parser = argparse.ArgumentParser()

    # Test options
    parser.add_argument("--test_only", action='store_true', default=True)

    # Save position
    parser.add_argument("--save_dir", type=str, default='./weakly_spuervisied_CD/',
                        help="path to Dataset")
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")

    # Datset Options
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: 2)")
    parser.add_argument("--label_path", type=str,
                        default='./weakly_spuervisied_CD/HCDNet/back_tar/',
                        help="path to Dataset txt file")
    parser.add_argument("--output_path", type=str,
                        default='./weakly_spuervisied_CD/HCDNet/back_oup/',
                        help="path to Dataset txt file")

    return parser

def normalization(data,max,min):
    _range = max-min
    return (data-min)/_range

def validate(opts, loader, metrics,threshold):

    metrics.reset()

    check = []
    for sample in loader:

        metrics.reset()
        print('---------------------------'+sample+'-----------------------------')
        target = np.load(opts.label_path + sample)
        pred_mask = np.load(opts.output_path +sample)
        metrics.update(target, pred_mask)

    score = metrics.get_results()
    return score


if __name__ == '__main__':
    opts = get_argparser().parse_args()

    test_dst = os.listdir(opts.label_path)
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    if opts.test_only:
        for threshold in [0.6]:
            print('threshold is:' + str(threshold))

            time_before_val = time.time()
            val_score = validate(opts=opts, loader=test_dst, metrics=metrics, threshold=threshold)
            time_after_val = time.time()

            print('Time_val = %f' % (time_after_val - time_before_val))

            print(metrics.to_str(val_score))
