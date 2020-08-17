import numpy as np
import matplotlib.pyplot as plt

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
        plt.show()
    else:
        ax.imshow(montage(np.array(sorted_centers)), vmin=vmin, vmax=vmax)
        plt.show()            
    counts = sorted(counts, key=lambda x:x[1], reverse=True)
    sorted_labels = np.zeros(len(labels))
    for i, entry in enumerate(counts):
        sorted_labels[labels == entry[0]] = i
    labels = sorted_labels

    plt.hist(labels, bins=len(centers))
    plt.show()

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
    fig = plt.figure(figsize=(30,30))
    n = len(hists)
    cols = 7
    rows = np.ceil(n/7)
    for i, hist in enumerate(hists):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.hist(hist, bins=180)
    plt.show()
        
