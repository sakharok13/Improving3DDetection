import time
import threading
from datetime import timedelta
from waymo_utils import WaymoOpenDataset
from common.utils import get_base_dir
from pathlib import Path

if __name__=="__main__":

    # training_dir = "/mnt/d/Projects/WaymoOpenDatasetToolKit/data/training" # provide directory where .tfrecords are stored
    training_dir = "/home/jovyan/data/waymo_open_dataset_v_1_3_2/lidar_raw_data"
    
    save_dir = Path(get_base_dir()) / '.output' / 'waymo' / 'processed'
    save_dir = save_dir.as_posix()

    toolkit = WaymoOpenDataset.ToolKit(training_dir=training_dir, save_dir=save_dir)

    for segment in toolkit.list_training_segments()[:1]:

        threads = []

        toolkit.assign_segment(segment)
        start = time.time()
        t1 = threading.Thread(target=toolkit.extract_camera_images)
        t2 = threading.Thread(target=toolkit.extract_laser_images)
        t1.start()
        t2.start()
        threads.append(t1)
        threads.append(t2)
        for thread in threads:
            thread.join()
        end = time.time()
        elapsed = end - start
        toolkit.save_video()
        toolkit.consolidate()
        print(timedelta(seconds=elapsed))
        break