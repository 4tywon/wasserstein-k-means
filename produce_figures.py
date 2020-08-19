import numpy as np
import matplotlib.pyplot as plt
import argparse
from skimage.util import montage
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_file_prefix', type=str, default='ribosome')
parser.add_argument('--experiment_name', type=str, required=True)
parser.add_argument('--outfile', type=str, required=True)

args = parser.parse_args()
data_file = args.data_file_prefix
name = args.experiment_name
ref_angles = np.load("data/" + data_file + "_angles.npy")
centers = np.load("experiment_runs/" + name + "-centers.npy")
labels = np.load("experiment_runs/" + name + "-labels.npy")
outfile = 'figures/' + args.outfile

if not os.path.exists('figures'):
    os.makedirs('figures')

def view_centers(labels, centers, vmin = None, vmax = None):
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
        plt.savefig(outfile + "-centers")
    else:
        ax.imshow(montage(np.array(sorted_centers)), vmin=vmin, vmax=vmax)
        plt.savefig(outfile + "-centers")
    counts = sorted(counts, key=lambda x:x[1], reverse=True)
    sorted_labels = np.zeros(len(labels))
    for i, entry in enumerate(counts):
        sorted_labels[labels == entry[0]] = i
    labels = sorted_labels
    plt.close(fig)
    plt.hist(labels, bins=len(centers))
    plt.savefig(outfile + "-occupancy")
    plt.close()

def view_angle_metrics(labels, rots):
    histograms = [cluster_histogram(i, rots, labels) for i in range(len(centers))]
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
    n = len(hists)
    h = 2 * (1+ int(n/7))
    fig = plt.figure(figsize=(h,30))

    cols = 7
    rows = max(4, np.ceil(n/7))
    for i, hist in enumerate(hists):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.hist(hist, bins=[i for i in range(180)])
    plt.savefig(outfile + "scaled-angle-histograms")
    plt.close(fig)
    fig = plt.figure(figsize=(h,30))
    for i, hist in enumerate(hists):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.hist(hist, bins=180)
    plt.savefig(outfile + "-angle-histograms")
    plt.close(fig)
    # plt.show()
        
view_centers(labels, centers)
view_angle_metrics(labels, ref_angles)
