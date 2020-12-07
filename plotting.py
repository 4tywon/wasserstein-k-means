import numpy as np
import matplotlib.pyplot as plt
import argparse
from skimage.util import montage
import os
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--outfile', type=str, required=True)
parser.add_argument('--snr', type=int, required=True)

args = parser.parse_args()
l2_centers = np.load("experiment_runs/l2-" + str(args.snr) + "-centered-centers.npy")
emd_centers = np.load("experiment_runs/wemd-"+ str(args.snr) + "-centered-centers.npy")

l2_labels = np.load("experiment_runs/l2-" + str(args.snr) + "-centered-labels.npy")
emd_labels = np.load("experiment_runs/wemd-" + str(args.snr) + "-centered-labels.npy")
outfile = 'figures/' + args.outfile

ref_angles = np.load("data/ribosome_angles_centered.npy")

DPI = 600

def save_figure(fig, name):
    filename = outfile + name
    print(f'Saving figure to "{os.path.realpath(filename)}"')
    fig.savefig(filename, dpi=DPI, bbox_inches='tight', format='pdf')

if not os.path.exists('figures'):
    os.makedirs('figures')

centers = l2_centers
labels = l2_labels
if labels is None:
    raise ValueError()
counts = dict()
for i in range(len(centers)):
    counts[i] = 0
print(len(centers))
for label in labels:
    counts[label] = counts[label] + 1
counts = [(label, counts[label]) for label in counts]
center_counts = [(centers[int(label[0])], label[1]) for label in counts]
sorted_centers = sorted(center_counts, key=lambda x:x[1], reverse=True)
sorted_centers = [c[0] for c in sorted_centers]
most_populated = sorted_centers[:8]
composite = np.zeros((2*128, 8*128))
for i, img in enumerate(most_populated):
    composite[:128, i*128:(i+1)*128] = img

centers = emd_centers
labels = emd_labels
if labels is None:
    raise ValueError()
counts = dict()
for i in range(len(centers)):
    counts[i] = 0
print(len(centers))
for label in labels:
    counts[label] = counts[label] + 1
counts = [(label, counts[label]) for label in counts]
center_counts = [(centers[int(label[0])], label[1]) for label in counts]
sorted_centers = sorted(center_counts, key=lambda x:x[1], reverse=True)
sorted_centers = [c[0] for c in sorted_centers]
most_populated = sorted_centers[:8]
for i, img in enumerate(most_populated):
    composite[128:2*128, i*128:(i+1)*128] = img

plt.imshow(composite, vmin=-1.8, vmax=3.9)
plt.xticks([])
plt.yticks([])
save_figure(plt, "-top_centers.pdf")
plt.close()

     
def view_angle_metrics(labels, rots, name):
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
    plot_aggregate(sorted_hists, name)

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
        for j in range(i + 1, len(rots)):
            angles.append(angle(rots[i], rots[j]))
    return angles

def plot_aggregate(hists, label):
    aggregate_hist = []
    for hist in hists:
        aggregate_hist += hist
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_bounds(0, 0.007)
    plt.yticks([])
    plt.xticks([30*i for i in range(7)], [str(30*i) + "$\degree$" for i in range(7)])
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xlim(0,180)
    if label =='$W_1$ Based Algorithm':
        sns.kdeplot(aggregate_hist, label=label, linewidth = 3)
    else:
        sns.kdeplot(aggregate_hist, label=label, linewidth = 3, linestyle='dashdot')

def plot_occupancy(labels, name):
    if labels is None:
        return
    counts = dict()
    for label in labels:
        counts[label] = counts[label] + 1 if label in counts else 1
    counts = [(label, counts[label]) for label in counts]
    counts = sorted(counts, key=lambda x:x[1], reverse=True)
    sorted_labels = np.zeros(len(labels))
    for i, entry in enumerate(counts):
        sorted_labels[labels == entry[0]] = i
    labels = sorted_labels
    plt.ylabel("Cluster Size", fontsize=16)
    plt.ylim(0,750)
    plt.yticks([75*i for i in range(11)])
    plt.xlim(0, 150)
    plt.xticks([25*i for i in range(7)])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xlabel("Rank of Cluster", fontsize=18)
    if name =='$W_1$ Based Algorithm':
        plt.hist(labels, bins = [i for i in range(150)], alpha=0.5, label=name)
    else:
        plt.hist(labels, bins = [i for i in range(150)], alpha=0.5, label=name)

plot_occupancy(l2_labels, '$L_2$ Based Algorithm')
plot_occupancy(emd_labels, '$W_1$ Based Algorithm')
if snr == 16:
    plt.legend(fontsize=20)
save_figure(plt, "-occupancy_dist-" + str(snr) + ".pdf")
plt.close()

view_angle_metrics(emd_labels, ref_angles, '$W_1$ Based Algorithm')
view_angle_metrics(l2_labels, ref_angles, '$L_2$ Based Algorithm')
if snr == 16:
    plt.legend(fontsize=20)
save_figure(plt, "-angle_dist-" + str(snr) + ".pdf")
plt.close()
