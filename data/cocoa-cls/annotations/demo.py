import json
import numpy as np
import os
import cv2
import pycocotools.mask as maskUtils


cats_of_interest = ['airplane', 'bicycle', 'bus', 'car', 'motorcycle', 'train']
img_dir = '../val2014/'

with open('COCO_amodal_val2014_with_classes.json', 'r') as fh:
    annotation = json.load(fh)

cats_indices = [None for i in cats_of_interest]
cats_counts = [0 for i in cats_of_interest]
cats_occ = [[] for i in cats_of_interest]

for cat in annotation['categories']:
    if cat['name'] in cats_of_interest:
        cats_indices[cats_of_interest.index(cat['name'])] = cat['id']

print(annotation['annotations'][0:10])
assert False
image_obj_pair = dict()

for ii, anno in enumerate(annotation['annotations']):
    if anno['category_id'] in cats_indices:
        try:
            image_obj_pair[anno['image_id']].append(ii)
        except:
            image_obj_pair[anno['image_id']] = [ii]
        cats_counts[cats_indices.index(anno['category_id'])] += 1
        cats_occ[cats_indices.index(anno['category_id'])].append(anno['occlude_rate'])



for i in range(len(cats_of_interest)):
    print('{}:      count - {}\n[FGL0 - {:.3f}, FGL1 - {:.3f}, FGL2 - {:.3f}, FGL3 - {:.3f}]'.format(cats_of_interest[i], cats_counts[i],
                                                                                   np.mean((np.array(cats_occ[i]) == 0).astype(int)),
                                                                                   np.mean((np.array(cats_occ[i]) <= 0.3).astype(int)) - np.mean((np.array(cats_occ[i]) == 0).astype(int)),
                                                                                   np.mean((np.array(cats_occ[i]) <= 0.6).astype(int)) - np.mean((np.array(cats_occ[i]) <= 0.3).astype(int)),
                                                                                   np.mean((np.array(cats_occ[i]) <= 1).astype(int)) - np.mean((np.array(cats_occ[i]) <= 0.6).astype(int))))

for image_anno in annotation['images']:
    try:
        objects = image_obj_pair[image_anno['id']]
    except:
        continue
    image_obj_pair[image_anno['id']] = {'file_name': image_anno['file_name'], 'objects': objects}

total_occ = [i for o in cats_occ for i in o]
print('TOTAL_NUM_FGL0:', np.sum((np.array(total_occ) == 0).astype(int)))
print('TOTAL_NUM_FGL1:', np.sum( (np.array(total_occ) <= 0.3).astype(int)) - np.sum((np.array(total_occ) == 0).astype(int)))
print('TOTAL_NUM_FGL2:', np.sum( (np.array(total_occ) <= 0.6).astype(int)) - np.sum((np.array(total_occ) <= 0.3).astype(int)))
print('TOTAL_NUM_FGL3:', np.sum( (np.array(total_occ) <= 1).astype(int)) - np.sum((np.array(total_occ) <= 0.6).astype(int)))


demo_dir = './demo/'
if not os.path.exists(demo_dir):
    os.mkdir(demo_dir)

done = False

for image_id in image_obj_pair.keys():

    objects = image_obj_pair[image_id]['objects']
    for obj_idx in objects:
        obj_anno = annotation['annotations'][obj_idx]
        inmodal_mask = maskUtils.decode(obj_anno['visible_mask'])[:, :, np.newaxis]

        try:
            occ_mask = maskUtils.decode(obj_anno['invisible_mask'])[:, :, np.newaxis]
        except:
            occ_mask = np.zeros(inmodal_mask.shape)

        amodal_mask = (inmodal_mask + occ_mask > 0.5).astype(int)

        occ = obj_anno['occlude_rate']

        category = cats_of_interest[cats_indices.index(obj_anno['category_id'])]

        img = cv2.imread(img_dir + image_obj_pair[image_id]['file_name'])
        cv2.imwrite(demo_dir + 'temp_inmodal{}.png'.format(obj_idx), img * inmodal_mask)
        cv2.imwrite(demo_dir + 'temp_amodal{}.png'.format(obj_idx), img * amodal_mask)

        print(category, occ, obj_anno['bbox'])


    assert False
