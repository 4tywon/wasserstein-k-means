import numpy as np
import logging
import os
import multiprocessing
import argparse
from tqdm import tqdm 
from scipy import ndimage
from skimage.transform import rescale

from image_ops import Dataset_Operations

logger = logging.getLogger('clustering')
logging.basicConfig(level = logging.INFO)

## Experiment Parameters

parser = argparse.ArgumentParser()
parser.add_argument('--snr', type=int, default=0)
parser.add_argument('--k', type=int,  default=100)
parser.add_argument('--n_angles', type=int,  default=200)
parser.add_argument('--niter', type=int, default=5)
parser.add_argument('--ncores', type=int, default=4)

parser.add_argument('--data_file_prefix', type=str, default='ribosome')
parser.add_argument('--clustering_type', type=str,  default='l2')
parser.add_argument('--experiment_name', type=str, required=True)


args = parser.parse_args()

# Global Variables
data_file = args.data_file_prefix
snr = args.snr
clustering_type = args.clustering_type
k = args.k
n_angles = args.n_angles
angles = [360/n_angles * i for i in range(n_angles)]
n_iter = args.niter
experiment_name = args.experiment_name
ncores = args.ncores
centers = None
labels = None

downsampled_resolution = 64

## Data Loading
def add_noise(images, snr):
    power_clean = (images**2).sum()/np.size(images)
    noise_var = power_clean/snr
    return images + np.sqrt(noise_var)*np.random.normal(0, 1, images.shape)
logger.info("Loading" + data_file + " data at snr level " + str(snr))
data = np.load("data/" + data_file + "_images.npy")
ref_angles = np.load("data/" + data_file + "_angles.npy")

if snr == 0:
    noisy_data = data
else:
    noisy_data = add_noise(data, snr)

noisy_data = noisy_data.astype('float32')
downsampling_ratio = downsampled_resolution / noisy_data.shape[1]
low_res_noise_data = np.array([rescale(im, downsampling_ratio) for im in noisy_data]).astype('float32')

logger.info("Beginning experiment ... ")
logger.info("Using " + clustering_type + " clustering with " + str(k) + " centers and " + str(n_angles) + "angles")
logger.info("Requested " + str(ncores) + " out of " + str(multiprocessing.cpu_count()))

image_dataset = Dataset_Operations(noisy_data, metric=clustering_type)
low_res_data = Dataset_Operations(low_res_noise_data, metric=clustering_type)
## Clustering Logic

def update_distance_for(center):
    dists = []
    for j, angle in enumerate(angles):
        dists.append(image_dataset.batch_distance_to(ndimage.rotate(center, angle, reshape=False)))
    return dists

def save():
    if not os.path.exists('experiment_runs'):
        os.makedirs('experiment_runs')
    centers_name = experiment_name + "-centers"
    labels_name = experiment_name + "-labels"
    np.save("experiment_runs/" + centers_name, centers)
    np.save("experiment_runs/" + labels_name, labels)


def initialize_centers(init='random_selected'):
    global centers
    global labels
    if centers is not None:
        return

    if init == 'random_selected':
        centers = []
        center_idxs = np.random.choice([i for i in range(image_dataset.n)], k, replace=False)
        for i in center_idxs:
            shape = image_dataset[i].shape
            centers.append(image_dataset[i])
        centers = np.array(centers)

    if init == 'k++':
        centers = _k_plus_plus()

def _k_plus_plus():
    chosen_centers_idx = [np.random.randint(low_res_data.n)]
    distances = np.zeros((low_res_data.n, len(chosen_centers_idx),len(angles)))
    for _ in tqdm(range(k-1)):
        if len(chosen_centers_idx) > 1:
            new_distances = np.zeros((low_res_data.n, len(chosen_centers_idx),len(angles)))
            new_distances[:, :len(chosen_centers_idx) - 1, :] = old_distances
            distances = new_distances
        for j, angle in enumerate(angles):
            dist = low_res_data.batch_distance_to(ndimage.rotate(low_res_data[chosen_centers_idx[-1]], angle, reshape=False))
            distances[:, -1, j] = dist
        old_distances = distances.copy()
        distances = distances.reshape(low_res_data.n, len(chosen_centers_idx) *len(angles))
        distances = distances.min(axis=1)**2
        distances[chosen_centers_idx] = 0
        probabilities = distances/distances.sum()
        probabilities = probabilities.reshape(image_dataset.n)
        next_center = np.random.choice(low_res_data.n, 1, probabilities.tolist())[0]
        chosen_centers_idx.append(next_center)
    centers = []
    for idx in chosen_centers_idx:
        centers.append(image_dataset[idx])
    return centers

def cluster(niter = 5, ncores = 1, init='random_selected'):
    global centers
    global labels
    initialize_centers(init=init)
    pool = multiprocessing.Pool(processes=ncores)

    for _ in tqdm(range(niter)):
        dists = np.array(pool.map(update_distance_for, centers))
        distances = np.transpose(dists, (2,0,1))
        min_distance_idxs = distances.reshape(image_dataset.n, k * len(angles)).argmin(axis=1)

        labels = np.floor(min_distance_idxs / len(angles))
        labels = labels
        orientations = np.array([angles[i] for i in min_distance_idxs % len(angles)])

        idxs = [[] for i in range(k)]
        orientation_lists = [[] for i in range(k)]
        for idx, label in enumerate(labels):
            label = int(label)
            idxs[label].append(idx)
            orientation_lists[label].append(orientations[idx])
        centers = image_dataset.batch_oriented_average(idxs, orientation_lists)
        save()

cluster(niter=n_iter, ncores=ncores)
