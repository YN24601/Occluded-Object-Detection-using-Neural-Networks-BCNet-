import json
import numpy as np


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


image_obj_pair = dict()

for ii, anno in enumerate(annotation['annotations']):
    if anno['category_id'] in cats_indices:
        try:
            image_obj_pair[anno['image_id']].append(ii)
        except:
            image_obj_pair[anno['image_id']] = [ii]
        cats_counts[cats_indices.index(anno['category_id'])] += 1
        cats_occ[cats_indices.index(anno['category_id'])].append(anno['occlude_rate'])

print(cats_occ)

for i in range(len(cats_of_interest)):
    print('{}:      count - {}\n[FGL0 - {:.3f}, FGL1 - {:.3f}, FGL2 - {:.3f}, FGL3 - {:.3f}]'.format(cats_of_interest[i], cats_counts[i],
                                                                                   np.mean((np.array(cats_occ[i]) == 0).astype(int)),
                                                                                   np.mean((np.array(cats_occ[i]) <= 0.3).astype(int)) - np.mean((np.array(cats_occ[i]) == 0).astype(int)),
                                                                                   np.mean((np.array(cats_occ[i]) <= 0.6).astype(int)) - np.mean((np.array(cats_occ[i]) <= 0.3).astype(int)),
                                                                                   np.mean((np.array(cats_occ[i]) <= 1).astype(int)) - np.mean((np.array(cats_occ[i]) <= 0.6).astype(int))))

total_occ = [i for o in cats_occ for i in o]
print('TOTAL_NUM_FGL0:', np.sum((np.array(total_occ) == 0).astype(int)))
print('TOTAL_NUM_FGL1:', np.sum( (np.array(total_occ) <= 0.3).astype(int)) - np.sum((np.array(total_occ) == 0).astype(int)))
print('TOTAL_NUM_FGL2:', np.sum( (np.array(total_occ) <= 0.6).astype(int)) - np.sum((np.array(total_occ) <= 0.3).astype(int)))
print('TOTAL_NUM_FGL3:', np.sum( (np.array(total_occ) <= 1).astype(int)) - np.sum((np.array(total_occ) <= 0.6).astype(int)))
