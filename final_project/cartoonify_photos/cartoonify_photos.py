#!/usr/bin/env python

# FINAL PROJECT
# Ngoc (Amy) Tran


""" Final Project - Building an Cartoon image

This cartoonify_photos will take the photos input and convert image to cartoon image

"""

import cv2
import numpy as np
import argparse
import time
from collections import defaultdict
from scipy import stats



def cartonify_image(image):
    """
    convert an inpuy image to a cartoon-like image
    Args:
       image: input PIL image

    Returns:
        out (numpy.ndarray): A grasycale or color image of dtype uint8, with
                             the shape of image
    """

    output = np.array(image)
    x, y, c = output.shape

    # noise removal while keeping edges sharp
    for i in xrange(c):
        output[:, :, i] = cv2.bilateralFilter(output[:, :, i], 5, 50, 50)

    #edges in an image using the Canny algorithm
    edge = cv2.Canny(output, 100, 200)
    #convert image into RGB color space
    output = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)

    #historygram array
    hists = []

    #Compute the histogram of a set of data.
    #H
    hist, _ = np.histogram(output[:, :, 0], bins=np.arange(180+1))
    hists.append(hist)
    #S
    hist, _ = np.histogram(output[:, :, 1], bins=np.arange(256+1))
    hists.append(hist)
    #V
    hist, _ = np.histogram(output[:, :, 2], bins=np.arange(256+1))
    hists.append(hist)

    centroids = []
    for h in hists:
        centroids.append(kmeans_histogram(h))
    print("centroids: {0}".format(centroids))

    output = output.reshape((-1, c))
    for i in xrange(c):
        channel = output[:, i]
        index = np.argmin(np.abs(channel[:, np.newaxis] - centroids[i]), axis=1)
        output[:, i] = centroids[i][index]
    output = output.reshape((x, y, c))
    output = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)

    # Retrieves contours from the binary image
    # RETR_EXTERNAL: retrieves only the extreme outer contours
    # CHAIN_APPROX_NONE= stores absolutely all the contour points
    contours, _ = cv2.findContours(edge,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    # Draws contours outlines
    cv2.drawContours(output, contours, -1, 0, thickness=1)
    return output


def update_centroids(centroids, hist):
    """
    This function perform update centroids until it can't change

    Args:
       centroids: centroids arrays
       hist: histogram arrays

    Returns:
        centroids, and groups

    """
    while True:
        groups = defaultdict(list)
        #assign pixel values
        for i in range(len(hist)):
            if hist[i] == 0:
                continue
            d = np.abs(centroids-i)
            index = np.argmin(d)
            groups[index].append(i)

        centroids_new = np.array(centroids)
        for i, indice in groups.items():
            if np.sum(hist[indice]) == 0:
                continue
            centroids_new[i] = int(np.sum(indice*hist[indice])/np.sum(hist[indice]))
        if np.sum(centroids_new-centroids) == 0:
            break
        centroids = centroids_new
    return centroids, groups


def kmeans_histogram(hist):
    """
    This is funtion choose the best K for k-means and get the centroids

    Args:
       hist: histogram arrays

    Returns:
        centroids

    """
    alpha = 0.001              # p-value threshold for normaltest
    min_size = 80                      # minimun group size for normaltest
    centroids = np.array([128])

    while True:
        centroids, groups = update_centroids(centroids, hist)

        #start increase K if possible
        centroids_new = set()     # use set to avoid same value when seperating centroid
        for i, indice in groups.items():
            #if there are not enough values in the group, do not seperate
            if len(indice) < min_size:
                centroids_new.add(centroids[i])
                continue

            # judge whether we should seperate the centroid
            # by testing if the values of the group is under a
            # normal distribution
            z, pval = stats.normaltest(hist[indice])
            if pval < alpha:
                #not a normal dist, seperate
                left = 0 if i == 0 else centroids[i-1]
                right = len(hist)-1 if i == len(centroids)-1 else centroids[i+1]
                delta = right-left
                if delta >= 3:
                    c1 = (centroids[i]+left)/2
                    c2 = (centroids[i]+right)/2
                    centroids_new.add(c1)
                    centroids_new.add(c2)
                else:
                    # though it is not a normal dist, we have no
                    # extra space to seperate
                    centroids_new.add(centroids[i])
            else:
                # normal dist, no need to seperate
                centroids_new.add(centroids[i])
        if len(centroids_new) == len(centroids):
            break
        else:
            centroids = np.array(sorted(centroids_new))
    return centroids

