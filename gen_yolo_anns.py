import os
import cv2
import csv
import random
from tqdm import tqdm
from collections import defaultdict

IMG_SIZE = 512
ORIGINAL_IMAGES = '/home/djordje/data/bee_track/frames'
ORIGINAL_LABELS = '/home/djordje/data/bee_track/frames_txt'
DST_IMGS_DIR = '/home/djordje/Documents/Projects/yudo/yolo_anns/images'
DST_ANN_DIR = '/home/djordje/Documents/Projects/yudo/yolo_anns/labels'
VAL_RATIO = 0.091

label_files = os.listdir(ORIGINAL_LABELS)
ann_30fps = defaultdict(list)
ann_70fps = defaultdict(list)

def read_label_file(label_filename):
    with open(label_filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')

        def parse_row(row):
            offset_x, offset_y = int(row[0]), int(row[1])
            bee_type = int(row[2]) - 1
            x, y = int(row[3]) / IMG_SIZE, int(row[4]) / IMG_SIZE
            angle = float(row[5])

            return offset_x, offset_y, bee_type, x, y, angle

        return list(map(parse_row, csv_reader))

def crop_and_save(data, txt_file_path):
    ''' Store crop images and txt files
    '''
    txt_list = []
    print('\nCrop and save...')
    for k, v in tqdm(data.items()):
        file_name, tl_x, tl_y = k.split('-')
        
        # Get crop image
        img_path = os.path.join(ORIGINAL_IMAGES, file_name + '.png')
        img = cv2.imread(img_path)
        crop = img[int(tl_y):int(tl_y) + 512, int(tl_x):int(tl_x) + 512, :]
        img_dst = os.path.join(DST_IMGS_DIR, k + '.png')
        cv2.imwrite(img_dst, crop)

        # Create yolo ann file
        ann_strings = []
        for r in v:
            ann_strings.append(' '.join(map(str, r)))

        txt_path = os.path.join(DST_ANN_DIR, k + '.txt')
        with open(txt_path, 'w') as f:
            f.write('\n'.join(ann_strings))

        txt_list.append(img_dst)
    
    # Save data part list of images
    with open(txt_file_path, 'w') as f:
        f.write('\n'.join(txt_list))

# Parse input annotations
for l_file in label_files:
    full_img_ann = read_label_file(os.path.join(ORIGINAL_LABELS, l_file))

    for line in full_img_ann:
        crop_name = f'{l_file.split(".txt")[0]}-{line[0]}-{line[1]}'

        # Split parsed labels according to fps
        if '_30fps_' in l_file:
            ann_30fps[crop_name].append(line[2:])
        else:
            ann_70fps[crop_name].append(line[2:])

# Split each part(fps) separately to the train/val and then merge
fps30_val_idx = int(VAL_RATIO * len(ann_30fps))
fps70_val_idx = int(VAL_RATIO * len(ann_70fps))

fps30_keys = list(ann_30fps.keys())
fps70_keys = list(ann_70fps.keys())
random.shuffle(fps30_keys)
random.shuffle(fps70_keys)

fps30_val_keys = fps30_keys[:fps30_val_idx]
fps30_train_keys = fps30_keys[fps30_val_idx:]
fps70_val_keys = fps70_keys[:fps70_val_idx]
fps70_train_keys = fps70_keys[fps70_val_idx:]

train_data = dict(zip(fps30_train_keys + fps70_train_keys,
                      [ann_30fps[k] for k in fps30_train_keys] + [ann_70fps[k] for k in fps70_train_keys]))
val_data = dict(zip(fps30_val_keys + fps70_val_keys,
                      [ann_30fps[k] for k in fps30_val_keys] + [ann_70fps[k] for k in fps70_val_keys]))

print(f'Length of training data: {len(train_data)}')
print(f'Length of validation data: {len(val_data)}')

crop_and_save(val_data, DST_ANN_DIR.replace('labels', 'val.txt'))
crop_and_save(train_data, DST_ANN_DIR.replace('labels', 'train.txt'))