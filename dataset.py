#import libraries
import shutil
import os
from PIL import Image
import PIL

#Create relevant directories for preprocessing if they don't exist
if not os.path.exists("/home/ubuntu/data/VLR/dataset/"):
    os.makedirs("/home/ubuntu/data/VLR/dataset/")
    print("Dataset Directory Created.")
if not os.path.exists("/home/ubuntu/data/VLR/dataset/test/"):
    os.makedirs("/home/ubuntu/data/VLR/dataset/test/")
if not os.path.exists("/home/ubuntu/data/VLR/dataset/train/"):
    os.makedirs("/home/ubuntu/data/VLR/dataset/train/")
if not os.path.exists("/home/ubuntu/data/VLR/dataset/test/blur/"):
    os.makedirs("/home/ubuntu/data/VLR/dataset/test/blur/")
if not os.path.exists("/home/ubuntu/data/VLR/dataset/test/sharp/"):
    os.makedirs("/home/ubuntu/data/VLR/dataset/test/sharp/")
if not os.path.exists("/home/ubuntu/data/VLR/dataset/train/blur/"):
    os.makedirs("/home/ubuntu/data/VLR/dataset/train/blur/")
if not os.path.exists("/home/ubuntu/data/VLR/dataset/train/sharp/"):
    os.makedirs("/home/ubuntu/data/VLR/dataset/train/sharp/")


test_source_folder = r"/home/ubuntu/data/VLR/raw_dataset/test/"
train_source_folder = r"/home/ubuntu/data/VLR/raw_dataset/train/"
test_destination_folder = r"/home/ubuntu/data/VLR/dataset/test/"
train_destination_folder = r"/home/ubuntu/data/VLR/dataset/train/"

sharp_test = []
sharp_train = []
blur_test = []
blur_train = []

i = 0

#Preprocessing Test files to newly created directories
print("Preprocessing Test Files.")
for file_name in os.listdir(test_source_folder):
    subfile = test_source_folder + file_name
    # print(subfile)
    for image_file in os.listdir(subfile):
        if image_file == 'blur':
            images_dir = os.path.join(subfile, image_file)
            for fname in os.listdir(images_dir):
                src_file  = os.path.join(images_dir, fname)
                new_dir = os.path.join(test_destination_folder, image_file)
                if (len(str(i))<6):
                    save_value = '0'*(6-len(str(i))) + str(i)
                dst_name = new_dir + '/' + save_value + '.png'
                os.rename(src_file, dst_name)
                i += 1

i = 0

for file_name in os.listdir(test_source_folder):
    subfile = test_source_folder + file_name
    # print(subfile)
    for image_file in os.listdir(subfile):
        if image_file == 'sharp':
            images_dir = os.path.join(subfile, image_file)
            for fname in os.listdir(images_dir):
                src_file  = os.path.join(images_dir, fname)
                new_dir = os.path.join(test_destination_folder, image_file)
                if (len(str(i))<6):
                    save_value = '0'*(6-len(str(i))) + str(i)
                dst_name = new_dir + '/' + save_value + '.png'
                os.rename(src_file, dst_name)
                i += 1

i = 0

for file_name in os.listdir(train_source_folder):
    subfile = train_source_folder + file_name
    # print(subfile)
    for image_file in os.listdir(subfile):
        if image_file == 'blur':
            images_dir = os.path.join(subfile, image_file)
            for fname in os.listdir(images_dir):
                src_file  = os.path.join(images_dir, fname)
                new_dir = os.path.join(train_destination_folder, image_file)
                if (len(str(i))<6):
                    save_value = '0'*(6-len(str(i))) + str(i)
                dst_name = new_dir + '/' + save_value + '.png'
                os.rename(src_file, dst_name)
                i += 1

i = 0

for file_name in os.listdir(train_source_folder):
    subfile = train_source_folder + file_name
    # print(subfile)
    for image_file in os.listdir(subfile):
        if image_file == 'sharp':
            images_dir = os.path.join(subfile, image_file)
            for fname in os.listdir(images_dir):
                src_file  = os.path.join(images_dir, fname)
                new_dir = os.path.join(train_destination_folder, image_file)
                if (len(str(i))<6):
                    save_value = '0'*(6-len(str(i))) + str(i)
                dst_name = new_dir + '/' + save_value + '.png'
                os.rename(src_file, dst_name)
                i += 1