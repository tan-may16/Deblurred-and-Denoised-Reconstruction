#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

import os
import argparse

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file",help="Input ROS bag.")
    parser.add_argument("output_dir",type = str,default = "/images/")
    parser.add_argument("image_topic",type = str, default = "/camera/color/image_raw")

    args = parser.parse_args()
    if args.output_dir == "/images/":
        if not os.path.exists("/images/"):
            os.makedirs("/images/")
            
    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    count = 0
    for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv_img = cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
        name = str(count)
        if len(name) < 6:
            name = '0' * (6 - len(name)) + name
        cv2.imwrite(os.path.join(args.output_dir, "%i.png" % name), cv_img)
        print ("Wrote image %i" % name)

        count += 1

    bag.close()

    return

if __name__ == '__main__':
    main()