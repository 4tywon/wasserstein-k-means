from scipy import ndimage
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import montage

class k_means:
    def __init__(self, k, n_angles):
        self.k = k
        self.angles = [360/n_angles * i for i in range(n_angles)]
        self.reset()

    def load(self, k, n_angles, centerfile, labelfile):
        self.k = k
        self.angles = [360/n_angles * i for i in range(n_angles)]
        self.centers = np.load(centerfile)
        self.labels = np.load(labelfile)

    def reset(self):
        self.centers = None
        self.evaluation_metrics = None
        self.labels = None

    def train(self, image_metric_space, niter = 5, init='random_selected'):

        self.initialize_centers(image_metric_space, init=init)

        for _ in range(niter):
            self.view_centers()
            distances = np.zeros((image_metric_space.n, self.k, len(self.angles)))
            print("Clustering")
            for i, center in enumerate(tqdm(self.centers)):
                for j, angle in enumerate(self.angles):
                    dist = image_metric_space.batch_distance_to(ndimage.rotate(center, angle, reshape=False))
                    distances[:, i, j] = dist
            min_distance_idxs = distances.reshape(image_metric_space.n, self.k * len(self.angles)).argmin(axis=1)

            labels = np.floor(min_distance_idxs / len(self.angles))
            self.labels = labels
            orientations = np.array([self.angles[i] for i in min_distance_idxs % len(self.angles)])

            idxs = [[] for i in range(self.k)]
            orientation_lists = [[] for i in range(self.k)]
            for idx, label in enumerate(labels):
                label = int(label)
                idxs[label].append(idx)
                orientation_lists[label].append(orientations[idx])
            print("Averaging")
            self.centers = image_metric_space.batch_oriented_average(idxs, orientation_lists)

    def initialize_centers(self, image_metric_space, init='random_selected'):
        if self.centers is not None:
            return

        if init == 'random_selected':
            centers = []
            center_idxs = np.random.choice([i for i in range(image_metric_space.n)], self.k, replace=False)
            for i in center_idxs:
                shape = image_metric_space[i].shape
                centers.append(image_metric_space[i] + np.abs(np.random.normal(0, 0.0001, shape)))
            self.centers = np.array(centers)

        if init == 'k++':
            self.centers = self._k_plus_plus(image_metric_space)

    def view_centers(self, vmin = None, vmax = None):
        labels = self.labels
        centers = self.centers
        if labels is None:
            return
        counts = dict()
        for label in labels:
            counts[label] = counts[label] + 1 if label in counts else 1
        counts = [(label, counts[label]) for label in counts]
        center_counts = [(centers[int(label[0])], label[1]) for label in counts]
        sorted_centers = sorted(center_counts, key=lambda x:x[1], reverse=True)
        sorted_centers = [c[0] for c in sorted_centers]
        fig, ax = plt.subplots(figsize=(30, 30))
        if vmin is None or vmax is None:
            ax.imshow(montage(np.array(sorted_centers)))
            plt.show()
        else:
            ax.imshow(montage(np.array(sorted_centers)), vmin=vmin, vmax=vmax)
            plt.show()            
        counts = sorted(counts, key=lambda x:x[1], reverse=True)
        sorted_labels = np.zeros(len(labels))
        for i, entry in enumerate(counts):
            sorted_labels[labels == entry[0]] = i
        labels = sorted_labels

        plt.hist(labels, bins=len(self.centers))
        plt.show()

    def view_angle_metrics(self, rots):
        labels = self.labels
        histograms = [cluster_histogram(i, rots, labels) for i in range(len(self.centers))]
        if labels is None:
            return
        counts = dict()
        for label in labels:
            counts[label] = counts[label] + 1 if label in counts else 1
        counts = [(label, counts[label]) for label in counts]
        hist_counts = [(histograms[int(label[0])], label[1]) for label in counts]
        sorted_hists = sorted(hist_counts, key=lambda x:x[1], reverse=True)
        sorted_hists = [h[0] for h in sorted_hists]
        plot_histograms(sorted_hists)


    def save(self, name):
        centers_name = name + "-centers"
        labels_name = name + "-labels"
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




def angle(R1, R2):
    dot = R1.T[:,2].T @ R2.T[:,2]
    if np.abs(dot) > 1:
        dot = dot/ np.abs(dot)
    theta = np.arccos(dot)* (180/np.pi)
    return theta

def cluster_histogram(i, rotations, labels):
    rots = rotations[labels == i]
    angles = []
    for i in range(len(rots)):
        for j in range(i, len(rots)):
            angles.append(angle(rots[i], rots[j]))
    return angles

def plot_histograms(hists):
    fig = plt.figure(figsize=(30,30))
    n = len(hists)
    cols = 7
    rows = np.ceil(n/7)
    for i, hist in enumerate(hists):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.hist(hist, bins=180)
    plt.show()
        