import numpy as np
import sys
import json
import glob
import os.path as osp
import cv2

def compute_whdr(reflectance, judgements, delta=0.1):
    points = judgements['intrinsic_points']
    comparisons = judgements['intrinsic_comparisons']
    id_to_points = {p['id']: p for p in points}
    rows, cols = reflectance.shape[0:2]

    error_sum = 0.0
    error_equal_sum = 0.0
    error_inequal_sum = 0.0

    weight_sum = 0.0
    weight_equal_sum = 0.0
    weight_inequal_sum = 0.0

    for c in comparisons:
        # "darker" is "J_i" in our paper
        darker = c['darker']
        if darker not in ('1', '2', 'E'):
            continue

        # "darker_score" is "w_i" in our paper
        weight = c['darker_score']
        if weight <= 0.0 or weight is None:
            continue

        point1 = id_to_points[c['point1']]
        point2 = id_to_points[c['point2']]
        if not point1['opaque'] or not point2['opaque']:
            continue

        # convert to grayscale and threshold
        l1 = max(1e-10, np.mean(reflectance[int(point1['y'] * rows), int(point1['x'] * cols), ...]))
        l2 = max(1e-10, np.mean(reflectance[int(point2['y'] * rows), int(point2['x'] * cols), ...]))

        # convert algorithm value to the same units as human judgements
        if l2 / l1 > 1.0 + delta:
            alg_darker = '1'
        elif l1 / l2 > 1.0 + delta:
            alg_darker = '2'
        else:
            alg_darker = 'E'

        if darker == 'E':
            if darker != alg_darker:
                error_equal_sum += weight
            weight_equal_sum += weight
        else:
            if darker != alg_darker:
                error_inequal_sum += weight
            weight_inequal_sum += weight

        if darker != alg_darker:
            error_sum += weight
        weight_sum += weight

    if weight_sum:
        return (error_sum / weight_sum), error_equal_sum/( weight_equal_sum + 1e-10), error_inequal_sum/(weight_inequal_sum + 1e-10)
    else:
        return None


#root = './testReal_cascade0_black_height120_width160/cascade0/iiw/'
root = 'IIW_cascade1/results_brdf2_brdf1/'
rootGt = '/home/zhl/CVPR20/Resubmission/Dataset/IIW/iiw-dataset/data/'
suffix = 'albedoBS1.png'

count = 0.0
whdr_sum = 0.0
whdr_mean = 0.0
img_list = glob.glob(osp.join(root, '*_%s' % suffix ) )

for img_path in img_list:
    #load CGI precomputed file
    judgement_path = osp.join(rootGt, img_path.split('/')[-1].split('_')[0] + '.json' )
    judgements = json.load(open(judgement_path) )


    count+=1.0
    ourR = cv2.imread(img_path ).astype(np.float32 ) / 255.0
    whdr, _, _ = compute_whdr(ourR, judgements )
    whdr_sum += whdr

    print('img_path: {0}, whdr: current {1} average {2}'.
            format(img_path.split('/')[-1].split('_')[0], whdr, whdr_sum / count ) )

whdr_mean = whdr_sum / count
print('whdr ours: {0}'.format(whdr_mean ) )
