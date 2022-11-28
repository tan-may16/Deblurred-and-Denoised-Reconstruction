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

#Preprocessing Test files to newly created directories
print("Preprocessing Test Files.")
for file_name in os.listdir(test_source_folder):
    subfile = test_source_folder + file_name
    # print(subfile)
    for image_file in os.listdir(subfile):
        if image_file == 'blur':
            images_dir = os.path.join(subfile, image_file)
            for fname in os.listdir(images_dir):
                # print(fname)
                # if fname not in test_blur:
                #     test_blur.append(fname)
                src_file  = os.path.join(images_dir, fname)
                dst_name = os.path.join(test_destination_folder, image_file)
                test_file =  os.path.join(dst_name, fname)
                if os.path.exists(test_file):
                    blur_test.append(src_file)
                    continue
                shutil.copy2(src_file, dst_name)

        elif image_file == 'sharp':
            images_dir = os.path.join(subfile, image_file)
            for fname in os.listdir(images_dir):
                src_file  = os.path.join(images_dir, fname)
                dst_name1 = os.path.join(test_destination_folder, image_file)
                test_file1 =  os.path.join(dst_name1, fname)
                if os.path.exists(test_file1):
                    sharp_test.append(src_file)
                    continue
                shutil.copy2(src_file, dst_name1)


for idx, images in enumerate(blur_test):
    save_value = 4100 + idx + 1
    save_value = '%02d' % save_value
    save_value = str(save_value)
    os.rename(images, '/home/ubuntu/data/VLR/dataset/test/blur/' + save_value + '.png')
    # picture = Image.open(images)
    # picture.save('/home/ubuntu/data/VLR/dataset/test/blur/' + save_value + '.png')

for idx, images in enumerate(sharp_test):
    save_value = 4100 + idx + 1
    save_value = '%02d' % save_value
    save_value = str(save_value)
    os.rename(images, '/home/ubuntu/data/VLR/dataset/test/sharp/' + save_value + '.png')
    # picture = Image.open(images)
    # picture.save('/home/ubuntu/data/VLR/dataset/test/sharp/' + save_value + '.png')

print("Copied Test Files.")

# print(len(blur_test))
# print(len(sharp_test))


#Preprocessing Train files to newly created directories
print("Preprocessing Train Files.")
for file_name in os.listdir(train_source_folder):
    subfile = train_source_folder + file_name
    # print(subfile)
    for image_file in os.listdir(subfile):
        if image_file == 'blur':
            images_dir = os.path.join(subfile, image_file)
            for fname in os.listdir(images_dir):
                src_file  = os.path.join(images_dir, fname)
                dst_name = os.path.join(train_destination_folder, image_file)
                test_file =  os.path.join(dst_name, fname)
                if os.path.exists(test_file):
                    blur_train.append(src_file)
                    continue
                shutil.copy2(src_file, dst_name)

        elif image_file == 'sharp':
            images_dir = os.path.join(subfile, image_file)
            for fname in os.listdir(images_dir):
                src_file  = os.path.join(images_dir, fname)
                dst_name1 = os.path.join(train_destination_folder, image_file)
                test_file =  os.path.join(dst_name1, fname)
                if os.path.exists(test_file):
                    sharp_train.append(src_file)
                    continue
                shutil.copy2(src_file, dst_name1)

for idx, images in enumerate(blur_train):
    save_value = 2900 + idx + 1
    save_value = '%02d' % save_value
    save_value = str(save_value)
    os.rename(images, '/home/ubuntu/data/VLR/dataset/train/blur/' + save_value + '.png')
    # picture = Image.open(images)
    # picture.save('/home/ubuntu/data/VLR/dataset/train/blur/' + save_value + '.png')

for idx, images in enumerate(sharp_train):
    save_value = 2900 + idx + 1
    save_value = '%02d' % save_value
    save_value = str(save_value)
    os.rename(images, '/home/ubuntu/data/VLR/dataset/train/sharp/' + save_value + '.png')
    # picture = Image.open(images)
    # picture.save('/home/ubuntu/data/VLR/dataset/train/sharp/' + save_value + '.png')

print("Copied Train Files.")

print("All Files Copied.")