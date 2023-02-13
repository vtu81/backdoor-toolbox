import os
import shutil

entries = sorted(item.name for item in os.scandir('data/imagenet/train/ILSVRC/Data/CLS-LOC/train'))

cnt = 0
tot = len(entries)

print('to re-orgnize %d files' % tot)


for entry in entries:

    dir_path = os.path.join('data/imagenet/train/ILSVRC/Data/CLS-LOC/train', entry)
    imgs_path = sorted(item.name for item in os.scandir(dir_path))
    dst_dir_path = os.path.join('data/imagenet/train/', entry)

    for img_path in imgs_path:
        src_file_path = os.path.join(dir_path, img_path)
        names = img_path.split('_')
        dst_file_path = os.path.join(dst_dir_path, names[1])
        shutil.move(src_file_path, dst_file_path)


    print('rebuild : ', entry)

print('done')