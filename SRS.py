"""
 This is the Python code of Spectral Region Splitting (SRS) method
 to reduce the dimensionality of hyperspectral images,
 written by E. Hosseini Aria

 The original article of the method is:
 Unsupervised dimensionality reduction of hyperspectral images using representations of reflectance spectra

"""

import numpy as np
from spectral import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# to read ENVI format images
import spectral.io.envi as envi
import time


def find_split_location(data1):
    min_err = [1000000000000.0, 0.0, 0.0, 0.0]
    n_chls = data1.shape[2]
    # rec_bandset = np.zeros((data1.shape[0], data1.shape[1], n_chls), dtype=float)
    for i in range(0, n_chls - 1):
        temp_band1 = np.mean(data1[:, :, 0:i + 1], axis=2)
        band1 = np.repeat(temp_band1[:, :, np.newaxis], i + 1, axis=2)
        rec_error1 = np.mean((np.power(data1[:, :, 0:i + 1] - band1, 2).sum(axis=2)))
        # rec_bandset[:, :, 0:i + 1] = np.repeat(temp_band1[:, :, np.newaxis], i + 1, axis=2)
        temp_band2 = np.mean(data1[:, :, i + 1: n_chls], axis=2)
        band2 = np.repeat(temp_band2[:, :, np.newaxis], n_chls - (i + 1), axis=2)
        rec_error2 = np.mean((np.power(data1[:, :, i + 1: n_chls] - band2, 2).sum(axis=2)))
        # rec_bandset[:, :, i+1: n_chls] = np.repeat(temp_band2[:, :, np.newaxis], n_chls-(i+1), axis=2)
        print(i)

        # plot the scan line
        # plt.clf()  # Clear the figure
        # plot_channels_vs_bands(img, split_location_array)
        # plt.plot(np.ones(3000) * i, list(range(3000)), 'g')
        # plt.pause(0.005)

        # the summation of the errors should be checked
        rec_error = rec_error1 + rec_error2
        # rec_error = (np.power(data1 - rec_bandset, 2).sum(axis=2))
        # rec_error = np.mean(np.sqrt(rec_error / n_chls))
        if rec_error < min_err[0]:
            min_err[0] = rec_error
            #  The real number of a channel i.e. array index+1
            min_err[1] = i + 1
            if rec_error1 == 0:
                rec_error1 = 0.000000001
            if rec_error2 == 0:
                rec_error2 = 0.000000001
            min_err[2] = rec_error1
            min_err[3] = rec_error2

    return min_err


""" 
This function plots the average spectral signal of the scene and 
the average spectral representation 

# input: original hyperspectral images, split locations, error of representation 
"""


def plot_channels_vs_bands(img, split_mat, err):
    if len(np.nonzero(split_mat)[0]):
        split_mat = np.sort(split_mat[np.nonzero(split_mat)])
        rec_bandset = np.zeros(img.shape, dtype=float)
        for i in range(0, len(np.nonzero(split_mat)[0])):
            if i == 0:
                temp_band = np.mean(img[:, :, 0:split_mat[i]], axis=2)
                rec_bandset[:, :, 0:split_mat[i]] = np.repeat(temp_band[:, :, np.newaxis], split_mat[i], axis=2)
            else:
                temp_band = np.mean(img[:, :, split_mat[i - 1]:split_mat[i]], axis=2)
                rec_bandset[:, :, split_mat[i - 1]:split_mat[i]] = np.repeat(temp_band[:, :, np.newaxis],
                                                                             split_mat[i] - split_mat[i - 1], axis=2)
            if i == (len(np.nonzero(split_mat)[0]) - 1):
                temp_band = np.mean(img[:, :, split_mat[i]:img.shape[2]], axis=2)
                rec_bandset[:, :, split_mat[i]:img.shape[2]] = np.repeat(temp_band[:, :, np.newaxis],
                                                                         img.shape[2] - split_mat[i], axis=2)
        band = np.mean(np.mean(rec_bandset, axis=0), axis=0)
        string = 'Number of bands = ' + str(len(np.nonzero(split_mat)[0]) + 1) + '  Representation error = ' \
                 + str(err[len(np.nonzero(split_mat)[0])])
    else:
        # When there is no split
        band = np.repeat(np.mean(np.mean(np.mean(img, axis=0), axis=0), axis=0), img.shape[2])
        string = 'Number of bands = 1' + 'Representation error = ' + str(err[0])

    plt.clf()  # Clear the figure
    gs = gridspec.GridSpec(1, 2)
    # Creates grid 'gs' of a rows and b columns
    ax = plt.subplot(gs[0, 0])
    # Adds subplot 'ax' in grid 'gs' at position [x,y]

    plt.plot(np.mean(np.mean(img, axis=0), axis=0), 'r', label='Original mean spectrum')
    plt.plot(band, 'b', label='Representation mean spectrum')
    plt.xlabel('Channel number')
    plt.ylabel('Average spectrum graph of the scene')
    plt.title('Representative spectral regions')
    plt.legend()
    ax2 = plt.subplot(gs[0, 1])
    ax2.set_ylabel('Error of representation (logarithmic)')
    ax2.set_xlabel('Number of bands')
    plt.suptitle(string)
    if len(np.nonzero(err)[0]) <= 15:
        dim = np.arange(1, len(np.nonzero(err)[0]) + 1, 1)
        plt.plot(dim, err[np.nonzero(err)], marker='o')
        plt.yscale('log')
    else:
        dim = np.arange(5, len(np.nonzero(err)[0]) + 1, 1)
        a = err[np.nonzero(err)[0]]
        plt.plot(dim, a[4:], marker='o')

start_time = time.time()


# Reading a hyperspectral image data set
#img = envi.open('195_bands.hdr', '195_bands').load()
header_adr='C:/Users/enaya/Documents/Hyperia/projects/hyperscout/Hyperia_datapackage/20181031_ceylanpinar/Hyperscout+S2A_BOA.hdr'
data_adr='C:/Users/enaya/Documents/Hyperia/projects/hyperscout/Hyperia_datapackage/20181031_ceylanpinar/Hyperscout+S2A_BOA'

img = envi.open(header_adr, data_adr).load()
img=img[:384,4:,:40]

# img = open_image('92AV3C.lan').load()
#img = img[122, 166,:]

plt.plot(np.mean(np.mean(img, axis=0), axis=0), 'r', label='Original mean spectrum')


rec_bandset = np.zeros(img.shape, dtype=float) # reconstruction band set
error_mat = np.zeros(img.shape[2], dtype=float)
split_location_array = np.zeros(100, dtype=int)
bandset_err = np.zeros(20, dtype=float)
plt_error_arr = np.zeros(30, dtype=float)
# img=open_image('92AV3c.lan').read_band(0)
# view = imshow(img, (10,10,10))

# to enable interactive plotting
plt.ion()
fig = plt.figure(figsize=(10, 5))
# Compute the error for one broad band and plot it
temp_band1 = np.mean(img[:, :, :], axis=2)
band1 = np.repeat(temp_band1[:, :, np.newaxis], img.shape[2], axis=2)
rec_error1 = np.mean((np.power(img[:, :, :] - band1, 2).sum(axis=2)))
plt_error_arr[0] = rec_error1
plot_channels_vs_bands(img, split_location_array, plt_error_arr)

for j in range(9):
    n_z = np.nonzero(split_location_array)
    if not len(n_z[0]):
        data1 = img
        min_err = find_split_location(data1)
        split_location_array[j] = min_err[1]
        plt_error_arr[1] = min_err[0]
        error_mat[j] = min_err[2]
        error_mat[j + 1] = min_err[3]
    else:
        sort_sla = np.sort(split_location_array[n_z])
        temp_min_error = np.zeros((len(sort_sla) + 1, 4))
        for k in range(0, len(n_z[0])):
            if (k == 0) and (sort_sla[k] != 1):
                data1 = img[:, :, 0:sort_sla[0]]
                min_err = find_split_location(data1)
                temp_min_error[0] = min_err
            elif (sort_sla[k] - sort_sla[k - 1] != 1) and (k != 0):
                data1 = img[:, :, sort_sla[k - 1]:sort_sla[k]]
                min_err = find_split_location(data1)
                min_err[1] = sort_sla[k - 1] + min_err[1]
                temp_min_error[k] = min_err
            if (k == (len(sort_sla)) - 1) and (img.shape[2] - sort_sla[k] != 1):
                data1 = img[:, :, sort_sla[k]:img.shape[2]]
                min_err = find_split_location(data1)
                min_err[1] = sort_sla[k] + min_err[1]
                temp_min_error[len(sort_sla)] = min_err
        min_rec_err = 1000000000000.0
        for k in range(0, len(np.nonzero(error_mat)[0])):
            rec_err = np.sum(error_mat) + np.sum(temp_min_error[k, 2:]) - error_mat[k]
            if rec_err < min_rec_err:
                min_rec_err = rec_err
                kk = k
                new_split = temp_min_error[kk, 1]
        split_location_array[j] = new_split
        # plot_channels_vs_bands(img, split_location_array)
        a = np.concatenate((temp_min_error[kk, 2:], error_mat[kk + 1:]), axis=None)
        error_mat[kk:] = a[:len(error_mat) - kk]
        plt_error_arr[j + 1] = np.sum(error_mat)
    plot_channels_vs_bands(img, split_location_array, plt_error_arr)
    plt.pause(5)
print("--- %s seconds ---" % (time.time() - start_time))
plot_channels_vs_bands(img, split_location_array, plt_error_arr)

# Y1 = img[10, 10]
# Y2 = img[100, 144]

# X=range(1,9900,45)
# view=imshow(img, (10,20,30))
""" plt.plot(Y1, 'r', Y2, 'b')
plt.axis([1, 221, 0, 10000])
plt.show()
# img=open_image('185_bands.tif')
# img=cv2.imread('185_bands.tif')
print(img.min)
# print(type (img))
# cv2.imshow('image', img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
"""
