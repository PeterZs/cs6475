# ASSIGNMENT 8
# Ngoc(Amy) Tran

import numpy as np
import scipy as sp
import scipy.signal
import cv2

# Import ORB as SIFT to avoid confusion.
try:
    from cv2 import ORB as SIFT
except ImportError:
    try:
        from cv2 import SIFT
    except ImportError:
        try:
            SIFT = cv2.ORB_create
        except:
            raise AttributeError("Your OpenCV(%s) doesn't have SIFT / ORB."
                                 % cv2.__version__)


""" Assignment 8 - Panoramas

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

def getImageCorners(image):
    """ For an input image, return its four corners.

    You should be able to do this correctly without instruction. If in doubt,
    resort to the testing framework. The order in which you store the corners
    does not matter.

    Note: The reasoning for the shape of the array can be explained if you look
    at the documentation for cv2.perspectiveTransform which we will use on the
    output of this function. Since we will apply the homography to the corners
    of the image, it needs to be in that format.

    Another note: When storing your corners, they are assumed to be in the form
    (X, Y) -- keep this in mind and make SURE you get it right.

    Args:
        image (numpy.ndarray): Input can be a grayscale or color image.

    Returns:
        corners (numpy.ndarray): Array of shape (4, 1, 2). Type of values in the
                                 array is np.float32.
    """
    corners = np.zeros((4, 1, 2), dtype=np.float32)
    # WRITE YOUR CODE HERE
    length, width = image.shape[:2]

    corners[0] = np.array([[0, 0]])
    corners[1] = np.array([[width, 0 ]])
    corners[2] = np.array([[0, length ]])
    corners[3] = np.array([[width, length]])

    return corners
    # END OF FUNCTION

def findMatchesBetweenImages(image_1, image_2, num_matches):
    """ Return the top list of matches between two input images.

    Note: You will not be graded for this function. This function is almost
    identical to the function in Assignment 7 (we just parametrized the number
    of matches). We expect you to use the function you wrote in A7 here. We will
    also release a solution for how to do this after A7 submission has closed.

    If your code from A7 was wrong, don't worry, you will not lose points in
    this assignment because your A7 code was wrong (hence why we will provide a
    solution for you after A7 closes).

    This function detects and computes SIFT (or ORB) from the input images, and
    returns the best matches using the normalized Hamming Distance through brute
    force matching.

    Args:
        image_1 (numpy.ndarray): The first image (grayscale).
        image_2 (numpy.ndarray): The second image. (grayscale).
        num_matches (int): The number of desired matches. If there are not
                           enough, return as many matches as you can.

    Returns:
        image_1_kp (list): The image_1 keypoints, the elements are of type
                           cv2.KeyPoint.
        image_2_kp (list): The image_2 keypoints, the elements are of type 
                           cv2.KeyPoint.
        matches (list): A list of matches, length 'num_matches'. Each item in 
                        the list is of type cv2.DMatch. If there are less 
                        matches than num_matches, this function will return as
                        many as it can.

    """
    # matches - type: list of cv2.DMath
    matches = None
    # image_1_kp - type: list of cv2.KeyPoint items.
    image_1_kp = None
    # image_1_desc - type: numpy.ndarray of numpy.uint8 values.
    image_1_desc = None
    # image_2_kp - type: list of cv2.KeyPoint items.
    image_2_kp = None
    # image_2_desc - type: numpy.ndarray of numpy.uint8 values.
    image_2_desc = None

    # COPY YOUR CODE FROM A7 HERE.
    #Initial SIFT detector
    orb = cv2.ORB()
    # find the keypoints and descriptors with SIFT
    image_1_kp, image_1_desc = orb.detectAndCompute(image_1,None)
    image_2_kp, image_2_desc = orb.detectAndCompute(image_2,None)
    # create BFMatcher object
    bfMacth =  cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bfMacth.match(image_1_desc, image_2_desc)
    # Sort the matches based on distances
    matches = sorted(matches, key = lambda x:x.distance)
    # return the top 'num_matches' matches in a list
    if len(matches) >= num_matches:
        matches = matches[:num_matches]

    return image_1_kp, image_2_kp, matches
  # END OF FUNCTION.

def findHomography(image_1_kp, image_2_kp, matches):
    """ Returns the homography between the keypoints of image 1, image 2, and
        its matches.

    Follow these steps:
        1. Iterate through matches and:
            1a. Get the x, y location of the keypoint for each match. Look up
                the documentation for cv2.DMatch. Image 1 is your query image,
                and Image 2 is your train image. Therefore, to find the correct
                x, y location, you index into image_1_kp using match.queryIdx,
                and index into image_2_kp using match.trainIdx. The x, y point
                is stored in each keypoint (look up documentation).
            1b. Set the keypoint 'pt' to image_1_points and image_2_points, it
                should look similar to this inside your loop:
                    image_1_points[match_idx] = image_1_kp[match.queryIdx].pt
                    # Do the same for image_2 points.

        2. Call cv2.findHomography and pass in image_1_points, image_2_points,
           use method=cv2.RANSAC and ransacReprojThreshold=5.0. I recommend
           you look up the documentation on cv2.findHomography to better
           understand what these parameters mean.
        3. cv2.findHomography returns two values, the homography and a mask.
           Ignore the mask, and simply return the homography.

    Args:
        image_1_kp (list): The image_1 keypoints, the elements are of type
                           cv2.KeyPoint.
        image_2_kp (list): The image_2 keypoints, the elements are of type 
                           cv2.KeyPoint.
        matches (list): A list of matches. Each item in the list is of type
                        cv2.DMatch.
    Returns:
        homography (numpy.ndarray): A 3x3 homography matrix. Each item in
                                    the matrix is of type numpy.float64.
    """
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)

    # WRITE YOUR CODE HERE.
    for i, match in enumerate(matches):
        image_1_points[i] = image_1_kp[match.queryIdx].pt
        image_2_points[i] = image_2_kp[match.trainIdx].pt

    homography, mask = cv2.findHomography(image_1_points, image_2_points, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    # Ignore the mask, and simply return the homography.
    return homography
    # END OF FUNCTION

def blendImagePair(warped_image, image_2, point):
    """ This is the blending function. We provide a basic implementation of
    this function that we would like you to replace.

    This function takes in an image that has been warped and an image that needs
    to be inserted into the warped image. Lastly, it takes in a point where the
    new image will be inserted.

    The current method we provide is very simple, it pastes in the image at the
    point. We want you to replace this and blend between the images.

    We want you to be creative. The most common implementation would be to take
    the average between image 1 and image 2 only for the pixels that overlap.
    That is just a starting point / suggestion but you are encouraged to use
    other approaches.

    Args:
        warped_image (numpy.ndarray): The image provided by cv2.warpPerspective.
        image_2 (numpy.ndarray): The image to insert into the warped image.
        point (numpy.ndarray): The point (x, y) to insert the image at.

    Returns:
        image: The warped image with image_2 blended into it.
    """
    # created an array of zeros  
    image_2_array_size = np.zeros(warped_image.shape, dtype=np.uint8)
    # ensure created a properly size 
    image_2_array_size[point[1]:image_2.shape[0]+point[1], \
                   point[0]:image_2.shape[1]+point[0]] = image_2

 
    #Convert images to GRAY color space
    warp_in_grey = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    image2_in_grey = cv2.cvtColor(image_2_array_size, cv2.COLOR_BGR2GRAY)
    # Compute the mask of two images using logical NOT
    warp_mask = 1 - np.logical_not(warp_in_grey)
    image2_mask = 1 - np.logical_not(image2_in_grey)
    # calculate union of the two images using logical AND
    union_mask = np.logical_and(warp_in_grey, image2_in_grey)

    # Use sum of union_mask to get the extents of uion
    col_sum = np.sum(union_mask, axis = 0)   
    idx_end = np.where(col_sum != 0)[0][-1]
    idx_start = np.where(col_sum != 0)[0][0]

    #Create a img_weight zero array image
    img_weight = np.zeros(warp_in_grey.shape, dtype=np.float32)

    for i, col in enumerate(union_mask.T[idx_start:idx_end]):
        img_weight[:,idx_start+i] = col * (1 - i / float(idx_end-idx_start))

    # mask for image_2
    image2_mask = image2_mask.astype(np.float32) - img_weight

    # Blend image
    output_image = np.zeros(warped_image.shape, dtype=np.uint8)
    if len(warped_image.shape) == 3:
        for i in range(warped_image.shape[2]):
            output_image[:,:,i] = (1 - image2_mask) * warped_image[:,:,i] + image2_mask * image_2_array_size[:,:,i]
    else:
        output_image = (1 - image2_mask) * warped_image + image2_mask * image_2_array_size

    return output_image
    # END OF FUNCTION

def warpImagePair(image_1, image_2, homography):
    """ Warps image 1 so it can be blended with image 2 (stitched).

    Follow these steps:
        1. Obtain the corners for image 1 and image 2 using the function you
        wrote above.
        
        2. Transform the perspective of the corners of image 1 by using the
        image_1_corners and the homography to obtain the transformed corners.
        
        Note: Now we know the corners of image 1 and image 2. Out of these 8
        points (the transformed corners of image 1 and the corners of image 2),
        we want to find the minimum x, maximum x, minimum y, and maximum y. We
        will need this when warping the perspective of image 1.

        3. Join the two corner arrays together (the transformed image 1 corners,
        and the image 2 corners) into one array of size (8, 1, 2).

        4. For the first column of this array, find the min and max. This will
        be your minimum and maximum X values. Store into x_min, x_max.

        5. For the second column of this array, find the min and max. This will
        be your minimum and maximum Y values. Store into y_min, y_max.

        6. Create a translation matrix that will shift the image by the required
        x_min and y_min (should be a numpy.ndarray). This looks like this:
            [[1, 0, -1 * x_min],
             [0, 1, -1 * y_min],
             [0, 0, 1]]

        Note: We'd like you to explain the reasoning behind multiplying the
        x_min and y_min by negative 1 in your writeup.

        7. Compute the dot product of your translation matrix and the homography
        in order to obtain the homography matrix with a translation.

        8. Then call cv2.warpPerspective. Pass in image 1, the dot product of
        the matrix computed in step 6 and the passed in homography and a vector
        that will fit both images, since you have the corners and their max and
        min, you can calculate it as (x_max - x_min, y_max - y_min).

        9. To finish, you need to blend both images. We have coded the call to
        the blend function for you.

    Args:
        image_1 (numpy.ndarray): Left image.
        image_2 (numpy.ndarray): Right image.
        homography (numpy.ndarray): 3x3 matrix that represents the homography
                                    from image 1 to image 2.

    Returns:
        output_image (numpy.ndarray): The stitched images.
    """
    # Store the result of cv2.warpPerspective in this variable.
    warped_image = None
    # The minimum and maximum values of your corners.
    x_min = 0
    y_min = 0
    x_max = 0
    y_max = 0

    # WRITE YOUR CODE HERE
    #*1.Obtain the corners for image 1 and image 2
    # using the function getImageCorners
    image_1_corners = getImageCorners(image_1)
    image_2_corners = getImageCorners(image_2)

    #*2.Transform the perspective of the corners of image 1
    image_1_corners = cv2.perspectiveTransform(image_1_corners, homography)

    #*3.Join the two corner arrays together
    joined = np.concatenate((image_1_corners, image_2_corners))

    #*4.For the first column of this array, find the min and max.
    # This willbe minimum and maximum X values.
    # Store into x_min, x_max.
    left = joined[:, :, :1]
    x_min = np.min(left)
    x_max = np.max(left)

    #*5.For the second column of this array, find the min and max.
    # This willbe the minimum and maximum Y values.
    # Store into y_min, y_max.
    right = joined[:, :, 1:]
    y_min = np.min(right)
    y_max = np.max(right)

    #*6.Create a translation matrix that will shift the image by the required
    #x_min and y_min (should be a numpy.ndarray)
    translation = np.array(
                        [[1, 0, -1 * x_min],
                        [0, 1, -1 * y_min],
                        [0, 0, 1]])
    #*7.Compute the dot product of your translation matrix & homography
    #use np.dor to product of two arrays.
    dotProduct = np.dot(translation, homography)

    #*8.Then call cv2.warpPerspective.
    vector = (x_max - x_min, y_max - y_min)
    warped_image = cv2.warpPerspective(image_1, dotProduct, vector)

    #*9.To finish, you need to blend both images.
    # We have coded the call to the blend function for you.

    # END OF CODING
    output_image = blendImagePair(warped_image, image_2,
                                  (-1 * x_min, -1 * y_min))
    return output_image

# Some simple testing.
#image_1 = cv2.imread("images/source/panorama_1/amy_1.jpg")
#image_2 = cv2.imread("images/source/panorama_1/amy_2.jpg")
#image_1 = cv2.imread("images/source/panorama_1/amy_1_2_result.jpg")
#image_2 = cv2.imread("images/source/panorama_1/amy_3.jpg")

#image_1_kp, image_2_kp, matches = findMatchesBetweenImages(image_1, image_2,
#                                                          20)
#homography = findHomography(image_1_kp, image_2_kp, matches)
#result = warpImagePair(image_1, image_2, homography)
#cv2.imwrite("images/output/amy_1_3_result.jpg", result)
