# -*- coding: utf-8 -*- 
import cv2
import os


def main():
    org_w_h = (2400, 1600)
    # scale_w_h = (480, 320) # multiple 32, and w:h = 3:2
    # scale_w_h = (192, 128)
    # scale_w_h = (384, 256)
    # scale_w_h = (576, 384)
    # scale_w_h = (672, 448)
    # scale_w_h = (768, 512)
    scale_w_h = (960, 640)
    # scale_w_h = (1920, 1280)

    if scale_w_h[0] % 32 !=0:
        print('you should w can be divided 32.')
        exit(1)

    if scale_w_h[1] % 32 !=0:
        print('you should h can be divided 32.')
        exit(1)

    ori_anno_txt = '/usr/local/wk/ys201810/od_word/datasets/object_detection/andon/train.txt'
    out_anno_txt = '/usr/local/wk/ys201810/od_word/datasets/object_detection/andon/train_' + str(scale_w_h[0]) + '_' + str(scale_w_h[1]) + '.txt'
    out_image_path = '/Volumes/Transcend/open_image_dataset_v4/image/all_image_resize_' + str(scale_w_h[0]) + '_' + str(scale_w_h[1]) + '/'

    if not os.path.exists(out_image_path):
        os.mkdir(out_image_path)

    print('org_size:' + str(org_w_h) + ' rescale:' + str(scale_w_h) + ' rate:' + str(scale_w_h[0] / org_w_h[0]))

    with open(out_anno_txt, 'a') as outf:
        with open(ori_anno_txt, 'r') as inf:
            for line in inf:
                out_str = ''
                line = line.rstrip()
                print('org line:', line)
                vals = line.split(' ')
                image_path = vals[0]
                obj_num = len(vals) - 1
                img = cv2.imread(image_path)
                img = cv2.resize(img, scale_w_h)

                out_image_path = image_path.replace('_' + str(org_w_h[0]) + '_' + str(org_w_h[1]), '_' + str(scale_w_h[0]) + '_' + str(scale_w_h[1]))
                # out_image_path = out_image_path + image_path.split('/')[-1].split('.')[0] + str(scale_w_h[0]) + '_' + str(scale_w_h[1]) + image_path.split('/')[-1].split('.')[1]

                out_str = out_image_path + ' '

                for i in range(obj_num):
                    annottations = vals[i + 1].split(',')
                    x1 = int(annottations[0])
                    y1 = int(annottations[1])
                    x2 = int(annottations[2])
                    y2 = int(annottations[3])
                    cate = annottations[4]

                    x1 = int(x1 * (scale_w_h[0] / org_w_h[0]))
                    x2 = int(x2 * (scale_w_h[0] / org_w_h[0]))
                    y1 = int(y1 * (scale_w_h[1] / org_w_h[1]))
                    y2 = int(y2 * (scale_w_h[1] / org_w_h[1]))

                    # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    if i == obj_num -1:
                        out_str = out_str + ','.join([str(x1), str(y1), str(x2), str(y2), cate]) + '\n'
                    else:
                        out_str = out_str + ','.join([str(x1), str(y1), str(x2), str(y2), cate]) + ' '

                # cv2.imshow("img", img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(out_image_path, img)

                outf.write(out_str)
                print('cng line:', out_str)


if __name__ == '__main__':
    main()