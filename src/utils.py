'''Finetune Stage'''

import os
import json


def convert_gt_to_coco(data_path, channel, image_path, field_name, output='coco.json'):
    '''Convert AWS Groud-Truth formatted json to COCO json'''
    D = {
        'images': [],
        'annotations': [],
        'categories': [
            {'id': 0, 'name': 'person'}
        ],
    }
    with open(os.path.join(data_path, f'{channel}.manifest')) as fp:
        anno_id = 0
        image_id = 0
        for line in fp.readlines():
            if not line.strip():
                continue

            info = json.loads(line.strip())
            labels = info[field_name]

            image_filepath = info['source-ref'][5:].rsplit('/', 1)[-1]
            # image_id = image_filepath.rsplit('/', 1)[1].split('.')[0]
            image_id += 1

            image = {
                'coco_url': os.path.join(image_path, image_filepath),
                'height': labels['image_size'][0]['height'],
                'width': labels['image_size'][0]['width'],
                'id': image_id,
            }
            D['images'].append(image)

            for bbox in labels['annotations']:
                anno = {
                    'category_id': bbox['class_id'],
                    'bbox': [
                        int(bbox['left']),
                        int(bbox['top']),
                        int(bbox['width']),
                        int(bbox['height'])
                    ],
                    'image_id': image_id,
                    'id': anno_id,
                    'iscrowd': 0,
                    'area': int(bbox['width']) * int(bbox['height']),
                }
                anno_id += 1
                D['annotations'].append(anno)

    with open(os.path.join(data_path, output), 'w') as out_fp:
        out_fp.write(json.dumps(D))


if __name__ == '__main__':
    from pycocotools.coco import COCO

    data_path = './'
    channel = 'train'
    image_path = '/'
    output = 'coco.json'
    convert_gt_to_coco(data_path, channel, image_path, 'labels', output)
    coco = COCO(os.path.join(data_path, output))
