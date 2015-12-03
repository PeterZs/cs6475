#!/usr/bin/env python
import cv2
import os
import time

from cartoonify_photos import cartonify_image


def test_generate_caroon_images():
    in_dir = './sources/input'
    out_dir = './sources/output'

    for f in os.listdir(in_dir):
        image = cv2.imread(os.path.join(in_dir, f))
        print('=====staring the convert cartoonify photo=====')
        print(f)
        start_time = time.time()
        output = cartonify_image(image)
        end_time = time.time()
        print("time: {0}s".format(end_time-start_time))
        name = os.path.basename(f)
        tmp = os.path.splitext(name)
        name = tmp[0]+"_cartoon" + tmp[1]
        name = os.path.join(out_dir, name)
        cv2.imwrite(name, output)

    print ("========DONE!=========")



if __name__ == "__main__":
    print 'Performing convert image to cartoonify images.'
    test_generate_caroon_images()

