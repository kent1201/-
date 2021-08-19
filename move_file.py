from pathlib import Path
import os
import random

random.seed(123)

def move(src, dst, size=0.1):
    fileList = os.listdir(src)
    random.shuffle(fileList)
    batch = round(1 / size)
    print("total images: , batch= ", len(fileList), batch)
    for i, item in enumerate(fileList):
        print("image name", item)
        if i % batch == 0:
            print("----------moved image name: ", item)
            Path(src+item).rename(dst+item)

print("------------------cnv---------------")
move("./dataset/cnv/", "./test/cnv/", 0.2)
print("------------------pcv---------------")
move("./dataset/pcv/", "./test/pcv/", 0.2)
