# ASSIGNMENT 4
# Ngoc(Amy) Tran

import cv2
import numpy as np
import scipy as sp

""" Assignment 4 - Detecting Gradients / Edges

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but the functions should NOT save the image to file. (This is a problem
    for us when grading because running 200 files results a lot of images being
    saved to file and opened in dialogs, which is not ideal). Thanks.

    2. DO NOT import any other libraries aside from the three libraries that we
    provide. You may not import anything else, you should be able to complete
    the assignment with the given libraries (and in most cases without them).

    3. DO NOT change the format of this file. Do not put functions into classes,
    or your own infrastructure. This makes grading very difficult for us. Please
    only write code in the allotted region.
"""

def imageGradientX(image):
    """ This function differentiates an image in the X direction.

    Note: See lectures 02-06 (Differentiating an image in X and Y) for a good
    explanation of how to perform this operation.

    The X direction means that you are subtracting columns:
    der. F(x, y) = F(x+1, y) - F(x, y)
    This corresponds to image[r,c] = image[r,c+1] - image[r,c]

    You should compute the absolute value of the differences in order to avoid
    setting a pixel to a negative value which would not make sense.

    We want you to iterate the image to complete this function. You may NOT use
    any functions that automatically do this for you.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        output (numpy.ndarray): The image gradient in the X direction. The shape
                                of the output array should have a width that is
                                one less than the original since no calculation
                                can be done once the last column is reached. 
    """

    # WRITE YOUR CODE HERE.
    gradientX = np.zeros((image.shape[0], image.shape[1]-1), dtype= np.float)

    for row in range (gradientX.shape[0]):
        for col in range (gradientX.shape[1]):
             gradientX[row,col] = abs( np.float(image[row,col+1]) - image[row,col] )

    gradientX = gradientX.astype (np.float)

    return gradientX
    # END OF FUNCTION.

def imageGradientY(image):
    """ This function differentiates an image in the Y direction.

    Note: See lectures 02-06 (Differentiating an image in X and Y) for a good
    explanation of how to perform this operation.

    The Y direction means that you are subtracting rows:
    der. F(x, y) = F(x, y+1) - F(x, y)
    This corresponds to image[r,c] = image[r+1,c] - image[r,c]

    You should compute the absolute value of the differences in order to avoid
    setting a pixel to a negative value which would not make sense.

    We want you to iterate the image to complete this function. You may NOT use
    any functions that automatically do this for you.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        output (numpy.ndarray): The image gradient in the Y direction. The shape
                                of the output array should have a height that is
                                one less than the original since no calculation
                                can be done once the last row is reached.
    """

    # WRITE YOUR CODE HERE.
    #image = image.astype(np.int32)
    #shape = image.shape
    gradientY = np.zeros((image.shape[0]-1, image.shape[1]),dtype= np.float)

    for row in range (gradientY.shape[0]):
        for col in range (gradientY.shape[1]):
             gradientY[row,col] = abs( np.float(image[row+1,col]) - image[row,col] )

    gradientY = gradientY.astype (np.float)

    return gradientY

    # END OF FUNCTION.

def computeGradient(image, kernel):
    """ This function applies an input 3x3 kernel to the image, and outputs the
    result. This is the first step in edge detection which we discussed in
    lecture.

    You may assume the kernel is a 3 x 3 matrix.
    View lectures 2-05, 2-06 and 2-07 to review this concept.

    The process is this: At each pixel, perform cross-correlation using the
    given kernel. Do this for every pixel, and return the output image.

    The most common question we get for this assignment is what do you do at
    image[i, j] when the kernel goes outside the bounds of the image. You are
    allowed to start iterating the image at image[1, 1] (instead of 0, 0) and
    end iterating at the width - 1, and column - 1.
    
    Note: The output is a gradient depending on what kernel is used.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.
        kernel (numpy.ndarray): A 3x3 kernel represented in a numpy array.

    Returns:
        output (numpy.ndarray): The computed gradient for the input image. The
                                size of the output array should be two rows and
                                two columns smaller than the original image
                                size.
    """

    # WRITE YOUR CODE HERE.
    assert kernel.shape[0] % 2 == 1, "Assume kernel 3x3 matrix! Must have odd number of rows"
    assert kernel.shape[1] % 2 == 1, "Assume kernel 3x3 matrix! must have odd number of columns"

    result = np.zeros((image.shape[0]-2, image.shape[1]-2), dtype= np.float)

    for row in range( 1, image.shape[0]-2 ):
        for col in range ( 1, image.shape[1]-2 ):
            image_correlation = image[ row-1:row+2, col-1:col+2 ]
            #result[row-1, col-1] = abs(np.sum( image_correlation * kernel))
            result[row-1, col-1] = abs(np.sum( np.multiply(image_correlation, kernel), axis=(0,1) ))

    result = result.astype (np.float)

    return result

    # END OF FUNCTION.


def convertToBlackAndWhite(image, threshold):

    #Return a new array of given shape and type, filled with zeros
    Black_White_Image = np.zeros(image.shape, dtype=np.uint8)
    # Multidimensional index iterator.
    for index, value in np.ndenumerate(image):
        # value > threshold, set white=255, else set 0= black
        if value > threshold:
            Black_White_Image[index] = 255  #white
        else:
            Black_White_Image[index] = 0  #black

    return Black_White_Image
