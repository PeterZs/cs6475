# ASSIGNMENT 2
# Ngoc(Amy) Tran


import cv2
import numpy as np
import scipy as sp

""" Assignment 2 - Basic Image Input / Output / Simple Functionality

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

def numberOfPixels(image):
    """ This function returns the number of pixels in a grayscale image.

    Note: A grayscale image has one dimension as covered in the lectures. You
    DO NOT have to account for a color image.

    You may use any / all functions to obtain the number of pixels in the image.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        int: The number of pixels in an image.
    """
    # WRITE YOUR CODE HERE.
    # image is grayscale,can caculate pixel by x*y (rows* columns)
    return image.size

    # END OF FUNCTION.

def averagePixel(image):
    """ This function returns the average color value of a grayscale image.

    Assignment Instructions: In order to obtain the average pixel, add up all
    the pixels in the image, and divide by the total number of pixels. We advise
    that you use the function you wrote above to obtain the number of pixels!

    You may not use numpy.mean or numpy.average. All other functions are fair
    game.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        int: The average pixel in the image (Range of 0-255).
    """
    # WRITE YOUR CODE HERE.
    return int(np.sum(image) / numberOfPixels(image))
    # END OF FUNCTION.

def convertToBlackAndWhite(image):
    """ This function converts a grayscale image to black and white.

    Assignment Instructions: Iterate through every pixel in the image. If the
    pixel is strictly greater than 128, set the pixel to 255. Otherwise, set the
    pixel to 0. You are essentially converting the input into a 1-bit image, as 
    we discussed in lecture, it is a 2-color image.

    You may NOT use any thresholding functions provided by OpenCV to do this.
    All other functions are fair game.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        numpy.ndarray: The black and white image.
    """
    # WRITE YOUR CODE HERE.
    #Return a new array of given shape and type, filled with zeros
    Black_White_Image = np.zeros(image.shape, dtype=np.uint8)
    # Multidimensional index iterator.
    for index, value in np.ndenumerate(image):
        # value > 128, set white=255, else set 0= black
        if value > 128:
            Black_White_Image[index] = 255  #white
        else:
            Black_White_Image[index] = 0  #black

    return Black_White_Image

    # END OF FUNCTION.

def averageTwoImages(image1, image2):
    """ This function averages the pixels of the two input images.

    Assignment Instructions: Obtain the average image by adding up the two input
    images on a per pixel basis and dividing them by two.

    You may use any / all functions to obtain the average image output.

    Note: You may assume image1 and image2 are the SAME size.

    Args:
        image1 (numpy.ndarray): A grayscale image represented in a numpy array.
        image2 (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        numpy.ndarray: The average of image1 and image2.

    """
    # WRITE YOUR CODE HERE.
    #Assume 2 image are the same size
    AvgImages = np.zeros(image1.shape, dtype=np.uint8)
    for index, value in np.ndenumerate(image1):
        AvgImages[index] = image1[index] / 2 + image2[index] / 2

    return AvgImages
    # END OF FUNCTION.

def flipHorizontal(image):
    """ This function flips the input image across the horizontal axis.

    Assignment Instructions: Given an input image, flip the image on the
    horizontal axis. This can be interpreted as switching the first and last
    column of the image, the second and second to last column, and so on.

    You may use any / all functions to flip the image horizontally.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        numpy.ndarray: The horizontally flipped image.

    """
    # WRITE YOUR CODE HERE.
    FlipImage = np.zeros(image.shape, dtype=np.uint8)
    for col in range(image.shape[1]):
        FlipImage[:,col] = image[:,image.shape[1]-col-1]

    return FlipImage
    # END OF FUNCTION.

#Main test codes
if False: ''' Comment out my test code
if __name__ == "__main__":
    image1 = cv2.imread("amy_test_image1.jpg", cv2.IMREAD_GRAYSCALE)
    # Writeback black and white pciture
    cv2.imwrite('amy_image1_BW.jpg', image1)
    image2 = cv2.imread("amy_test_image2.jpg",cv2.IMREAD_GRAYSCALE)
    # 1) print pixel from grayscale images
    print "\n=>(1)numberOfPixels: image1  "
    print numberOfPixels(image1)
    print "\n---print image1.shape to test:--"
    print image1.shape
    print "\n=>**(1)numberOfPixels: image2  "
    print numberOfPixels(image2)
    print "\n---print image2.shape to test:--"
    print image2.shape

    # 2) print average pixel from grayscale images
    print "\n=>(2)averagePixel: image1  "
    print averagePixel(image1)
    print "\n=>**(2)averagePixel: image2  "
    print averagePixel(image2)
    print "\n---use another method to test by print np.average:--"
    print np.average(image1)

    #3)call converToBlackAndWhite and store that output
    print "\n=>(3)convertToBlackAndWhite...  "
    bwImage1= convertToBlackAndWhite (image1)
    print "\n+++Write the Black and White image... please wait!  "
    cv2.imwrite('amy_image1_convertBW.jpg', bwImage1)

    #4) The averageTwoImages function
    print "\n=>(4)averageTwoImages(image1,image2) "
    average = averageTwoImages(image1, image2)
    print "\n++++Write the averageTwoImage ... please wait!  "
    cv2.imwrite("amy_average.jpg", average)

     #5) Now, flip the image horizontal
    print "\n=>(5)flipHorizontal-----  "
    imageFlip = flipHorizontal(image1)
    print "\n+++Write the flipHorizontal() image ... please wait!  "
    cv2.imwrite('amy_test_image1_flip.jpg', imageFlip)
'''



