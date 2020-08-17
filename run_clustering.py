import numpy as np
import logging
import os
import multiprocessing
import argparse


from image_ops import Dataset_Operations
from clustering import k_means

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

data_file = args.data_file_prefix
snr = args.snr
clustering_type = args.clustering_type
k = args.k
n_angles = args.n_angles
n_iter = args.niter
experiment_name = args.experiment_name
ncores = args.ncores

def add_noise(images, snr):
    power_clean = (images**2).sum()/np.size(images)
    noise_var = power_clean/snr
    return images + np.sqrt(noise_var)*np.random.normal(0, 1, images.shape)

data = np.load("data/" + data_file + "_images.npy")
angles = np.load("data/" + data_file + "_angles.npy")

if snr == 0:
    noisy_data = data
else:
    noisy_data = add_noise(data, snr)

logger.info("Loaded " + data_file + " data at snr level " + str(snr))

logger.info("Beginning experiment ... ")

logger.info("Using " + clustering_type + " clustering with " + str(k) + " centers and " + str(n_angles) + "angles")

logger.info("Requested " + str(ncores) + " out of " + str(multiprocessing.cpu_count()))

dataset = Dataset_Operations(noisy_data, metric=clustering_type)
clustering = k_means(k, n_angles, experiment_name=experiment_name)
clustering.train(dataset, niter=n_iter, init='k++', ncores=ncores)


