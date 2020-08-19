import numpy as np
from scipy import ndimage, signal, optimize
import pywt
from tqdm import tqdm

class Dataset_Operations:
    def __init__(self, images, metric='wemd', level=6):
        self.images = mask(images, images[0].shape)
        self.metric = metric
        if metric == 'wemd':
            self.wavelet_space = wave_transform_volumes(self.images, level)
        self.n = self.images.shape[0]
        self.level = level

    def batch_distance_to(self, image):
        if self.metric == 'l2':
            distances = [np.linalg.norm(self.images[i] - image) for i in range(len(self.images))]
            return np.array(distances)

        elif self.metric == 'wemd':
            wavelet_img =  wave_transform_volumes([image], self.level)[0]
            distances = [np.abs(self.wavelet_space[i] - wavelet_img).sum() for i in range(len(self.wavelet_space))]
            return np.array(distances)

    def distance(self, i, j):
        if self.metric == 'l2':
            return np.sqrt(((self.images[i] - self.images[j])**2).sum())

        elif self.metric == 'wemd':
            return np.abs(self.wavelet_space[i] - self.wavelet_space[j]).sum()

    def batch_oriented_average(self, idxs, orientation_lists, strategy='mean', ncores = 1, **kwargs):
        ''' orientations will be opposite the direction that the image will be rotated in'''
        rotated_image_sets = []
        for idx, orientations in zip(idxs, orientation_lists):
            rotated_images = [ndimage.rotate(im, -1 * orientations[i], reshape=False) for i, im in enumerate(self.images[idx])]
            rotated_image_sets.append(np.asarray(rotated_images))

        if strategy == 'mean':
            centers = []
            for rotated in rotated_image_sets:
                if rotated.shape[0] != 0:
                    centers.append(np.sum(rotated, axis=0) / rotated.shape[0])
                else:
                    centers.append(np.ones(self.images[0].shape))
            return np.asarray(centers)

        elif strategy == 'emd-bary':
            reg = kwargs['reg']
            numItermax = 15000 if 'numItermax' not in kwargs else kwargs['numItermax']
            return [barycenter(rotated, reg, numItermax=numItermax) for rotated in rotated_image_sets]

    def __getitem__(self, idx):
        return self.images[idx]

    def __getslice__(self, i, j):
        return self.images[i:j]


## Computing wEMD

DIMENSION = 2
WAVELET_NAME = 'sym5'

def wave_emd(p1,p2):
    p = np.asarray(p1)-np.asarray(p2)
    p = np.abs(p)
    emd = np.sum(p)
    return emd


def volume_to_wavelet_domain(volume, level, wavelet=pywt.Wavelet(WAVELET_NAME)):
    """
    This function computes an embedding of non-negative 3D Numpy arrays such that the L_1 distance
    between the resulting embeddings is approximately equal to the Earthmover distance of the arrays.
    It implements the weighting scheme in Eq. (20) of the Technical report by Shirdhonkar, Sameer, and David W. Jacobs. "CAR-TR-1025 CS-TR-4908 UMIACS-TR-2008-06." (2008).
    """
    assert len(volume.shape) == DIMENSION

    volume_dwt = pywt.wavedecn(volume, wavelet, mode='zero', level=level)

    detail_coefs = volume_dwt[1:]
    n_levels = len(detail_coefs)

    weighted_coefs = []
    for (j, details_level_j) in enumerate(volume_dwt[1:]):
        for coefs in details_level_j.values():
            multiplier = 2**((n_levels-1-j)*(1+(DIMENSION/2.0)))
            weighted_coefs.append(coefs.flatten()*multiplier)

    return np.concatenate(weighted_coefs)


def wave_transform_volumes(volumes, level):
    wavelet = pywt.Wavelet(WAVELET_NAME)
    return [volume_to_wavelet_domain(vol, level, wavelet) for vol in volumes]


## Computing Barycenter

def barycenter(A, reg, numItermax=15000,
                               stopThr=1e-9, stabThr=1e-300, verbose=False,
                               mode='dot'):

    weights = np.ones(A.shape[0]) / A.shape[0]
    H_0 = np.max([_entropy(i) for i in A] + [_entropy(np.sum(A, axis=0))])

    b = np.zeros_like(A[0, :, :])
    U = np.ones_like(A)
    KV = np.ones_like(A)

    cpt = 0
    err = 1

    # build the convolution operator
    
    if mode == 'dot':
        t = np.linspace(0, 1, A.shape[1])
        [Y, X] = np.meshgrid(t, t)
        xi2 = np.exp(-(X - Y) ** 2 / reg)
        def K(x):
            return np.dot(np.dot(xi2, x), xi2)
    elif mode == 'fft':
        std = 1/reg
        kernlen =25
        gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
        gkern2d = np.outer(gkern1d, gkern1d)
        def K(x):
            return signal.fftconvolve(x, gkern2d, mode='same')

    while (err > stopThr and cpt < numItermax):
        bold = b
        cpt = cpt + 1

        b = np.zeros_like(A[0, :, :])
        for r in range(A.shape[0]):
            KV[r, :, :] = K(A[r, :, :] / np.maximum(stabThr, K(U[r, :, :])))
            b += weights[r] * np.log(np.maximum(stabThr, U[r, :, :] * KV[r, :, :]))
        b = np.exp(b)
        b = b**_entropic_sharpening(b, H_0) 
        for r in range(A.shape[0]):
            U[r, :, :] = b / np.maximum(stabThr, KV[r, :, :])

        if cpt % 10 == 1:
            err = np.sum(np.abs(bold - b))

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
                
    print(cpt)

    return b

    
def _entropic_sharpening(b, H_0):
    sol = optimize.root_scalar(_entropy_eq(b, H_0), bracket = [0, 30], method='bisect', maxiter=50)
    return sol.root if sol.converged else 1

def _entropy(im):
    dist = im/im.sum()
    dist_no_zero = np.copy(dist)
    dist_no_zero[dist_no_zero <= 0] = 1
    return -np.sum(dist*np.log(dist_no_zero))

def _entropy_eq(im, H_0):
    dist = im/im.sum()
    def f(beta):
        return np.sum(dist**beta) + _entropy(dist**beta) - (1 + H_0)
    return f


def mask(data, size):
    size = size[0]
    mask = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if ((i - ((size-1)/2))**2 + (j - ((size-1)/2))**2 <= ((size-1)/2)**2):
                mask[i,j] = 1
    return data * mask


