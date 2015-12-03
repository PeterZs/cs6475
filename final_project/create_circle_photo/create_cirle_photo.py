# FINAL PROJECT
# Ngoc (Amy) Tran


""" Final Project -put your image into cluster of circle

Create an image into clusters of circles, with radius determined by pixel brightness.

"""
from __future__ import division
import sys
import Image
import ImageDraw
import argparse
import numpy as np
from math import sqrt

# Progress Indicator lets you visualize the
# progress of a programming task 
ENABLE_PROGRESS_INDICATOR = True
try:
    import pyprind
except ImportError:
    ENABLE_PROGRESS_INDICATOR = False

#Debug log to print message
ENABLE_LOG = False


def log_msg (message):
    """
    print log message
    Args:
       message: message to print out

    Returns:
        none
    """
    global ENABLE_LOG
    if ENABLE_LOG:
        print message
        sys.stdout.flush()


def open_image(image, scale=1.0, grey=True):
    """
    Open an image with params and can resize image as scale value  
    Args:
       image: input of image
       scale : scale to resize image
       

    Returns:
        image_file : A grasycale or color image of dtype uint8, with
                     the shape of image
    """
    try:
        log_msg("Opening image: %s" % image)
        image_file = Image.open(image)
    except Exception as e:
        error_msg = ("Image file you provided:\n{image}\ndoes not exist! Here's what the computer"
                     "says:\n{exception}".format(image=image, exception=e))
        sys.exit(error_msg)

    if scale != 1.0:
        image_file = image_file.resize(tuple(int(i * scale) for i in image_file.size))

    if grey:
        # convert image to monochrome
        image_file = image_file.convert('L')
    return image_file


def image_overlapping(c1, c2):
    # circle data type:
    # (x, y, rad)
    dist = sqrt( (c2[0] - c1[0])**2 + (c2[1] - c1[1])**2 )
    if c1[2] + c2[2] > dist:
        return True
    return False


def image_render(circles, path, params, imsize):
    """
    Open an image with params and can resize image as scale value  
    Args:
       circles: input of image     
       path : image path
       params : params input image 
       imsize : image size
       
    Returns:        
      none
    """
    log_msg("Image Rendering...")

    if params['bgimg']:
        bg = open_image(params['bgimg'], grey=False)
        bgim = bg.resize(imsize)
        bgpix = bgim.load()

    col = params['bgcolour']
    col = 255 if col > 255 else col
    col = 0 if col < 0 else col
    bgcolour = (col, col, col)

    outline = (0, 0, 0)
    if params['nooutline']:
        outline = None
    
    final = Image.new('RGB', imsize, bgcolour)
    draw = ImageDraw.Draw(final)

    image_x, image_y = imsize
    
    for y in range(image_y):
        for x in range(image_x):
            circle_radius = circles[x,y]
            if circle_radius != 0:
                bbox = (x - circle_radius, y - circle_radius, 
                      x + circle_radius, y + circle_radius)
                fill = bgpix[x, y] if params['bgimg'] else (255, 255, 255)
                draw.ellipse(bbox, fill=fill, outline=outline)
    del draw
    final.save(params['outimg'])


def create_circlephoto(params):
    """
    put image into clusters of circles with radius determined by pixel brightness.
  
    Args:
       params : params of create cirle photo image       
    """
    global ENABLE_LOG
    global ENABLE_PROGRESS_INDICATOR

    interval = params['interval']
    maxrad = params['maxrad']
    scale = params['scale']
    
    img_file = open_image(params['circimg'], scale)
    
    pixels = img_file.load()
    circles = np.zeros(img_file.size, int)

    """ 
    For each pixel in the original image, determine its
    "grey" brightness, and determine an appropriate radius
    for that.
    In the local region for other circles (local is 
    determined by the max_radius of other circles + the
    radius of the current potential circle).

    * If there is some circles nearby, check to see 
       if the new circle will overlap with it or not. 
    
    *If all nearby circles won't overlap, then record 
     the radius in a 2D array that corresponds to the image.
    """
    image_x, image_y = img_file.size
    skips = 0

    if ENABLE_LOG  and ENABLE_PROGRESS_INDICATOR :
        progress = pyprind.ProgBar(image_y, stream=1)

    for y in range(0, image_y, interval):
        prev_rad = 0
        closeness = 0
        for x in range(0, image_x, interval):
            closeness += 1

            # Determine radius
            greyval = pixels[x, y]
            radius = int(maxrad * (greyval/255))
            if radius == 0:
                radius = 1

            # If we are still going to be inside the last circle
            # placed on the same X row, save time and skip.
            if prev_rad + radius >= closeness:
                skips += 1
                continue
            # Define bounding box.
            bbox = [x - radius - maxrad, 
                  y - radius - maxrad, 
                  x + radius + maxrad, 
                  y + radius + maxrad]

            
            if bbox[0] < 0:       # Ensure the bounding box is OK with 
                bbox[0] = 0       # edges. We don't need to check the 
            if bbox[1] < 0:       # outer edges because it's OK for the
                bbox[1] = 0       # centre to be right on the edge.
            if bbox[2] >= image_x:
                bbox[2] = image_x - 1
            if bbox[3] >= image_y:
                bbox[3] = image_y - 1
            
            c1 = (x, y, radius)
            
            # Use bounding box and numpy to extract the local area around the
            # circle. Then use numpy to do a boolean operating to give a 
            # true/false matrix of whether circles are nearby.
            local_area = circles[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            circle_nearby = local_area != 0
           
            coords_of_local_circles = np.where(circle_nearby)
            # Need the extra dim for next step
            radii_of_local_cirles = np.expand_dims(local_area[circle_nearby], axis=0)  
            nrby_cirles = np.vstack([coords_of_local_circles, radii_of_local_cirles])
            nrby_cirles = nrby_cirles.transpose()

            any_overlaps_here = False
            if nrby_cirles.shape[0] == 0:
                circles[x,y] = radius
                prev_rad = radius
                closeness = 0
            else:
                for n in nrby_cirles:
                    c2 = (n[0]+bbox[0], n[1]+bbox[1], n[2]) 
                    overlap = image_overlapping(c1, c2)        
                    if overlap:
                        any_overlaps_here = True
                        break
                # Look if any nearby circles overlap. If any do, don't make
                # a circle here.
                if not any_overlaps_here:               
                    circles[x, y] = radius 
                    prev_rad = radius
                    closeness = 0
        if ENABLE_LOG is True and ENABLE_PROGRESS_INDICATOR is True:
            progress.update()

    log_msg("Avoided {skips} calculations".format(skips=skips))

    image_render(circles, "", params, img_file.size)


def main(argv=None):
    """
    Open an image with params and can resize image as scale value  
    Args:
       image: input of image
       scale : scale to resize image
       

    Returns:
        image_file : A grasycale or color image of dtype uint8, with
                     the shape of image
    """
    parser = argparse.ArgumentParser(description="Using create_circle_photo!")

    addarg = parser.add_argument # just for cleaner code
    
    addarg("--circimg", type=str, required=True,
            help="The image that will make up the circles.", )

    addarg("--interval", type=int, default=1, 
            help="Interval between pixels to look at in the circimg. 1 means all pixels.")
    
    addarg("--bgimg", type=str, 
            help="An image to colour the circles with. Will be resized as needed.")
    
    addarg("--outimg", type=str, required=True,
            help="Filename for the outputted image.")
    
    addarg("--maxrad", type=int, default=10,
            help="Max radius of a circle (corresponds to a white pixel)")
    
    addarg("--scale", type=float, default=1,
            help="Percent to scale up the circimg (sometimes makes it look better).")

    addarg("--bgcolour", type=int, default=255,
            help="Grey-scale val from 0 to 255")

    addarg("--nooutline", action='store_true', default=False,
            help="When specified, no outline will be drawn on circles.")

    addarg("--log", action='store_true', default=False,
            help="Write progress to stdout.")
    
    parsed_args = parser.parse_args()
    params = dict(parsed_args.__dict__)

    global ENABLE_LOG
    if params["log"] is True:
        ENABLE_LOG = True

    log_msg("Starting Create Circle Photo...")
    create_circlephoto(params)


if __name__ == "__main__":
    sys.exit(main())
