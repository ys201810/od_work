# -*- coding: utf-8 -*- 
import os
import json
from PIL import Image, ImageDraw
import cv2

def main():
    json_file = '../../dataset/object_detection/andon/json/all_image_resize_2400_1600.json'
    image_dir = '/Volumes/Transcend/open_image_dataset_v4/image/all_image_resize_2400_1600/'
    image_wbb_dir = '/Volumes/Transcend/open_image_dataset_v4/image/all_image_resize_2400_1600_with_bb/'

    with open(json_file, 'r') as inf:
        json_d = json.load(inf)

    org_w = 2400
    org_h = 1600

    total_num = 0

    with open('../../dataset/object_detection/andon/train.txt', 'a') as outf:

        for key in json_d['frames'].keys():
            out_str = ''
            # print(key, type(json_d['frames'][key]), len(json_d['frames'][key]) ,json_d['frames'][key])
            img = cv2.imread(image_dir + key)
            # drew = ImageDraw.Draw(img)
            bb_num = len(json_d['frames'][key])

            print(key, bb_num, json_d['frames'][key])
            if bb_num == 0:
                continue

            out_str = image_dir + key + ' '

            for i in range(bb_num):

                x1 = int(json_d['frames'][key][i]['x1'])
                x2 = int(json_d['frames'][key][i]['x2'])
                y1 = int(json_d['frames'][key][i]['y1'])
                y2 = int(json_d['frames'][key][i]['y2'])
                width = int(json_d['frames'][key][i]['width'])
                height = int(json_d['frames'][key][i]['height'])
                tags = json_d['frames'][key][i]['tags']
                total_num += 1

                if tags[0] == 'andon':
                    category_num = 1

                x1 = int(x1 * org_w / width)
                x2 = int(x2 * org_w / width)
                y1 = int(y1 * org_h / height)
                y2 = int(y2 * org_h / height)
                # drew.rectangle([(x1, y1), (x2, y2)], outline=(0, 0, 0))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 7)

                if i == bb_num - 1:
                    out_str = out_str + ','.join([str(x1), str(y1), str(x2), str(y2), str(category_num)]) + '\n'
                else:
                    out_str = out_str + ','.join([str(x1), str(y1), str(x2), str(y2), str(category_num)]) + ' '

            # cv2.imshow("img", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite(image_wbb_dir + key, img)
            outf.write(out_str)

    print(total_num)


if __name__ == '__main__':
    main()