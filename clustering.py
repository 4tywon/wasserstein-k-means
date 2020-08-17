from scipy import ndimage
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import montage

class k_means:
    def __init__(self, k, n_angles, experiment_name):
        self.k = k
        self.angles = [360/n_angles * i for i in range(n_angles)]
        self.reset()
        self.name = experiment_name

    def load(self, k, n_angles, centerfile, labelfile):
        self.k = k
        self.angles = [360/n_angles * i for i in range(n_angles)]
        self.centers = np.load(centerfile)
        self.labels = np.load(labelfile)

    def reset(self):
        self.centers = None
        self.evaluation_metrics = None
        self.labels = None

    def train(self, image_dataset, niter = 5, ncores = 1, init='random_selected'):
        self.initialize_centers(image_dataset, init=init)

        def update_distance_for(center):
            dists = []
            for j, angle in enumerate(self.angles):
                dist.append(image_dataset.batch_distance_to(ndimage.rotate(center, angle, reshape=False)))
            return dists

        pool = multiprocessing.Pool(processes=ncores)

        for _ in range(niter):
            dists = np.array(pool.map(update_distance_for, self.centers))
            distances = np.transpose(distances, (2,0,1))
            min_distance_idxs = distances.reshape(image_dataset.n, self.k * len(self.angles)).argmin(axis=1)

            labels = np.floor(min_distance_idxs / len(self.angles))
            self.labels = labels
            orientations = np.array([self.angles[i] for i in min_distance_idxs % len(self.angles)])

            idxs = [[] for i in range(self.k)]
            orientation_lists = [[] for i in range(self.k)]
            for idx, label in enumerate(labels):
                label = int(label)
                idxs[label].append(idx)
                orientation_lists[label].append(orientations[idx])
            self.centers = image_dataset.batch_oriented_average(idxs, orientation_lists)

    def initialize_centers(self, image_dataset, init='random_selected'):
        if self.centers is not None:
            return

        if init == 'random_selected':
            centers = []
            center_idxs = np.random.choice([i for i in range(image_dataset.n)], self.k, replace=False)
            for i in center_idxs:
                shape = image_dataset[i].shape
                centers.append(image_dataset[i] + np.abs(np.random.normal(0, 0.0001, shape)))
            self.centers = np.array(centers)

        if init == 'k++':
            self.centers = self._k_plus_plus(image_dataset)

    def save(self):
        centers_name = self.name + "-centers"
        labels_name = self.name + "-labels"
        np.save(centers_name, self.centers)

    def _k_plus_plus(self, image_space):
        chosen_centers_idx = [np.random.randint(image_space.n)]
        distances = np.zeros((image_space.n, len(chosen_centers_idx),len(self.angles)))
        for _ in tqdm(range(self.k-1)):
            if len(chosen_centers_idx) > 1:
                new_distances = np.zeros((image_space.n, len(chosen_centers_idx),len(self.angles)))
                new_distances[:, :len(chosen_centers_idx) - 1, :] = old_distances
                distances = new_distances
            for j, angle in enumerate(self.angles):
                dist = image_space.batch_distance_to(ndimage.rotate(image_space[chosen_centers_idx[-1]], angle, reshape=False))
                distances[:, -1, j] = dist
            old_distances = distances.copy()
            distances = distances.reshape(image_space.n, len(chosen_centers_idx) *len(self.angles))
            distances = distances.min(axis=1)**2
            distances[chosen_centers_idx] = 0
            probabilities = distances/distances.sum()
            probabilities = probabilities.reshape(image_space.n)
            next_center = np.random.choice(image_space.n, 1, probabilities.tolist())[0]
            chosen_centers_idx.append(next_center)
        print(chosen_centers_idx)
        centers = []
        for idx in chosen_centers_idx:
            centers.append(image_space[idx])
        return centers