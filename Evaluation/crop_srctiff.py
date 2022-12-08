import numpy as np
import os
import skimage.io as skio



if __name__ == '__main__':
    image_path = './GF1_datasets/GF1_WFV_dataset/tiff/'
    save_path = './GF1_datasets/GF1_WFV_dataset/JPEGImages/'
    # txt_path = './GF1_datasets/ImageSets/train.txt'

    os.makedirs(save_path, exist_ok=True)

    cut_h = 321
    cut_w = 321
    overlap_pix =60

    # dst = []
    # with open(txt_path, 'r') as gf:
    #     for lines in gf.readlines():
    #         c = lines.replace('\n', '').split('_')
    #         name = c[0]+'_'+c[1]
    #         if name not in dst:
    #             dst.append(name)
    # print('%d train patches. \n' % (len(dst)))

    ALL_tiff = os.listdir(image_path)

    for files in ALL_tiff:
        print('----------------------------------Producing ' + files + '----------------------------------')

        content = files.replace('.tiff','').split('_')
        src_name = content[-2] + '_' + content[-1]
        #
        # if src_name not in dst:
        #     continue

        num = 0
        rsData = skio.imread(image_path + files, plugin="tifffile")
        row, col, channel = rsData.shape[0], rsData.shape[1], rsData.shape[2]
        print('Original image info:  %sx%sx%s ' % (str(row), str(col), str(channel)))

        try:
            if cut_h <= row or cut_w <= col:
                # vertical direction
                cut_h_nums = row // (cut_h - overlap_pix) + 1
                # horizontal direction
                cut_w_nums = col // (cut_w - overlap_pix) + 1
                
                print('Cut into: %sx%s'% (str(cut_h_nums), str(cut_w_nums)))
                

                for vertical in range(cut_h_nums):
                    for horizontal in range(cut_w_nums):
                        # start line/w
                        Lc = (horizontal * cut_w) - overlap_pix * horizontal
                        up_r = (vertical * cut_h) - overlap_pix * vertical

                        if Lc <= 0:
                            Lc = 0
                        if up_r <= 0:
                            up_r = 0

                        Rc = Lc + cut_w
                        down_r = up_r + cut_h

                        # print('\t\tLc=%d   Rc=%d' % (Lc, Rc))
                        # print('\t\tup=%d   do=%d' % (up_r, down_r))

                        if Rc >= col:
                            Rc = col
                        if down_r >= row:
                            down_r = row

                        if Lc==Rc or up_r==down_r:
                            continue

                        visCrop = rsData[up_r:down_r, Lc:Rc, :]


                        name_string = str(src_name) \
                                      + '_' + str(Lc) + '_' + str(Rc) + '_' + str(up_r) + '_' + str(down_r) \
                                      + '_' + str(num)

                        if os.path.exists(save_path+name_string+'.npy'):
                            print('File exist!')
                            continue

                        if (visCrop.shape[0] < cut_h) or (visCrop.shape[1] < cut_w):
                            continue;
                        else:
                            np.save(save_path + name_string + '.npy', visCrop)
                            print(visCrop.shape)
                            print(np.max(visCrop))

            else:
                print('Origin file less than crop_size')

        except OSError:
            print('No Free place!!!')