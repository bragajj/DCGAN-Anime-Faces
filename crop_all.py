from PIL import Image
import os.path, sys

path = "train/"
dirs = os.listdir(path)
#Crop all pictures down to 64 x 64
# This is equivalent to the main idle frame
def crop():
    itemCount = 0
    for item in dirs:
        itemCount += 1
        fullpath = os.path.join(path,item)      
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            f, e = os.path.splitext(fullpath)
            imCrop = im.crop((0, 0, 64, 64))
            imCrop.save(f + '.png', "PNG", quality=100)

crop()
print("Done")

