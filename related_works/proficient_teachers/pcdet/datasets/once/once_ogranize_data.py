import os
from shutil import copy

if __name__ == '__main__':

    dataset_root = "/mnt/junbo/datasets/once/3D_lidars/data"
    others = ['3D_images', '3D_infos']
    for scene in sorted(os.listdir(dataset_root)):
        target_path = os.path.join(dataset_root, scene)
        source_path = target_path.replace("3D_lidars", others[-1])
        metadata = os.path.join(source_path, "{}.json".format(scene))
        if os.path.isfile(metadata):
            copy(metadata, target_path)

