import numpy as np
import os
import skimage.io as skio
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cv2

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    src_path = '/home/FAKEDATA/GF1_datasets/bigpatch/bigpatch_mask/'
    mask_path = '/home/liuyang/weakly_spuervisied_CD/PHCNet/oup/'
    label_path = '/home/FAKEDATA/GF1_datasets/datasets_321/data/SegmentationClass/'
    vis_path = '/home/liuyang/weakly_spuervisied_CD/PHCNet/back_oup_vis/'
    save_path = '/home/liuyang/weakly_spuervisied_CD/PHCNet/back_oup/'

    os.makedirs(vis_path,exist_ok=True)
    os.makedirs(save_path,exist_ok=True)

    num,folder_num=0,0

    cut_h = 321
    cut_w = 321
    ol = 60

    # get all origin tiff names

    txt_path = '/home/FAKEDATA/GF1_datasets/datasets_321/data/ImageSets/test_MFC.txt'

    thumb_list = []
    with open(txt_path, 'r') as gf:
        for lines in gf.readlines():
            c = lines.split('_')
            thumb_list.append(c[0]+'_'+c[1]+'_'+c[2])
            # thumb_list.append(lines.replace('\n',''))



    for thumb in thumb_list:

        if os.path.exists(save_path + str(thumb) + '_back.npy'):
            continue

        # rsData = skio.imread(src_path + thumb + '.tiff',plugin="tifffile")
        content = thumb.split('_')
        folder = content[0]+'_'+content[1]
        rsData = np.load(src_path + folder+'/'+thumb + '.npy')
        src_h,src_w = rsData.shape[0],rsData.shape[1]
        backmap = torch.zeros((src_h,src_w),dtype=torch.long)

        folder_num+=1


        for files in os.listdir(mask_path):
            content = files.replace('_2oup.npy','').split('_')

            src_file = content[0] + '_' + content[1]+ '_' + content[2]

            if src_file != thumb:
                continue

            print(str(folder_num)+'/'+str(num)+'----------------------------------Producing ' + files + '----------------------------------')
            num+=1


            crop_data = np.load(mask_path + files)
            crop_data = np.column_stack((crop_data, crop_data[:, -1]))
            crop_data = np.row_stack((crop_data, crop_data[-1, :]))
            crop_data = torch.tensor(crop_data)

            crop_data = crop_data.to(device)
            backmap = backmap.to(device)

            Lc = int(content[-4])
            Rc = int(content[-3])
            Ur = int(content[-2])
            Dr = int(content[-1])
            Mc = Lc + ol
            Mr = Ur + ol

            if Rc-Lc<=ol or Dr-Ur<=ol:
                continue

            # block1   block2
            # block3   block4
            print('Source   : %sx%s' % (str(src_h),str(src_w)))
            print('Position : %sx%sx%s' % (str(Ur),str(Mr),str(Dr)))
            print('Position : %sx%sx%s' % (str(Lc),str(Mc),str(Rc)))
            print()

            if (Dr-Ur==cut_h) and (Rc-Lc==cut_w):
                backmap[Ur:Dr, Lc:Rc] = backmap[Ur:Dr, Lc:Rc] + crop_data

        joint = backmap[:-cut_h, :-cut_w]
        joint[joint > 0] = 1
        joint = joint.cpu().numpy()
        np.save(save_path + str(thumb) + '_back.npy', joint)

        fig, ax = plt.subplots()
        ax.imshow(joint, cmap='gray', aspect='equal')
        plt.axis('off')
        height, width = joint.shape
        fig.set_size_inches(width / 100.0, height / 100.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)

        plt.savefig(vis_path + str(thumb) + '_back.png')
        plt.close()


