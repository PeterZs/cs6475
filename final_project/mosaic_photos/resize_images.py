# FINAL PROJECT
# Ngoc (Amy) Tran


""" Final Project - This is the Utility to resize images

The mosaic_photos will take the photos input as square!
So, I use this utility to resize the images

1) Put all the images to folder "source/input"
2) the resize image output as "source/output"

"""

from PIL import Image

import cv2
import os
import sys
import time


input_path = "./source/input/"
input_dir = os.listdir( input_path )
output_path = "./source/output/"
output_dir = os.listdir( output_path )
idx = 0

def resize_images():
    for item in input_dir:
      
      
        if os.path.isfile(input_path+item):
            print('=====staring resize photo=====')
            print(item)
            global idx;
            idx += 1
	    print ("---output image number----")
            print (idx)
            infile = Image.open(input_path+item)           
            filename, e = os.path.splitext(input_path+item)
            start_time = time.time()
            imResize = infile.resize((500,500), Image.ANTIALIAS)   
            #imResize.save(output_path +item,'JPEG', quality=90)	   
            imResize.save(output_path +'pix{0:02d}.jpg'.format(idx),'JPEG', quality=90)           
            end_time = time.time()
            print("==>time: {0}s".format(end_time-start_time))
    print ("========DONE!=========")

if __name__ == "__main__":
    print 'Performing resize images.'
    resize_images()





