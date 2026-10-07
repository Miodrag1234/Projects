import tensorflow as tf
import numpy as np
from collections import OrderedDict
import os
import glob
import cv2

rng = np.random.RandomState(2017)

def np_load_frame(filename, resize_height, resize_width):
    image_decoded = cv2.imread(filename)
    if resize_height != None and resize_width != None:
        image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    else:
        image_resized = image_decoded
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized

     
        
def np_load_size(filename):
    image_decoded = cv2.imread(filename)
    height = image_decoded.shape[0] 
    width = image_decoded.shape[1]  
    size = np.array([width, height], dtype=np.float32)
    return np.expand_dims(size, 1)

class DataLoader(object):
    def __init__(self, data_folder):
        self.dir = data_folder
        self.datas = OrderedDict()
        self.flat_pairs = None
        self.setup()

    def __call__(self, batch_size):
        if self.flat_pairs is not None:
            length = len(self.flat_pairs)
        else:
            data_info_list = list(self.datas.values())
            length = data_info_list[0]['length']

        def data_clip_generator():
            #frame_id = 0
            while True:
                data_clip = []
                size_clip = []
                frame_id = rng.randint(0, length-1)
                #######inputs
                if self.flat_pairs is not None:
                    left_path, right_path = self.flat_pairs[frame_id]
                    data_clip.append(np_load_frame(left_path, 128, 128))
                    data_clip.append(np_load_frame(right_path, 128, 128))
                    size_clip.append(np_load_size(left_path))
                else:
                    data_clip.append(np_load_frame(data_info_list[0]['frame'][frame_id], 128, 128))
                    data_clip.append(np_load_frame(data_info_list[1]['frame'][frame_id], 128, 128))
                    size_clip.append(np_load_size(data_info_list[0]['frame'][frame_id]))
                data_clip = np.concatenate(data_clip, axis=2)
                #######size
                size_clip = np.concatenate(size_clip, axis=0)
                
                yield (data_clip, size_clip)

        dataset = tf.data.Dataset.from_generator(generator=data_clip_generator, output_types=(tf.float32, tf.float32),
                                                  output_shapes=([128, 128, 6], [2,1]))
        print('generator dataset, {}'.format(dataset))
        dataset = dataset.prefetch(buffer_size=1000)
        dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
        print('epoch dataset, {}'.format(dataset))

        return dataset

    def __getitem__(self, data_name):
        assert data_name in self.datas.keys(), 'data = {} is not in {}!'.format(data_name, self.datas.keys())
        return self.datas[data_name]

    def setup(self):
        # Prihvaca:
        # 1) dvije podmape s .jpg (input1/input2), ili
        # 2) flat sekvencu slika u jednom folderu (uparivanje 1-2, 3-4, ...)
        all_items = glob.glob(os.path.join(self.dir, '*'))
        subdirs_with_jpg = []
        for item in all_items:
            if not os.path.isdir(item):
                continue
            jpgs = glob.glob(os.path.join(item, '*.jpg'))
            if jpgs:
                subdirs_with_jpg.append(item)
        subdirs_with_jpg.sort()
        if len(subdirs_with_jpg) >= 2:
            for i, data in enumerate(subdirs_with_jpg[:2]):
                name = 'input1' if i == 0 else 'input2'
                self.datas[name] = {}
                self.datas[name]['path'] = data
                self.datas[name]['frame'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[name]['frame'].sort()
                self.datas[name]['length'] = len(self.datas[name]['frame'])
            print('DataLoader (subfolders):', list(self.datas.keys()))
            return

        flat_jpg = glob.glob(os.path.join(self.dir, '*.jpg'))
        flat_jpg.sort()
        if len(flat_jpg) < 2:
            raise ValueError(
                'TEST_FOLDER mora sadrzavati podmape input1/input2 ili barem 2 .jpg u root folderu. '
                'Pronadeno podmapa: {}, root jpg: {}'.format(
                    [os.path.basename(d) for d in subdirs_with_jpg], len(flat_jpg)
                )
            )

        pair_count = len(flat_jpg) // 2
        if pair_count == 0:
            raise ValueError('Nije moguce formirati parove iz sekvence slika.')
        if len(flat_jpg) % 2 != 0:
            print('Upozorenje: neparan broj slika u folderu, zadnja ce biti ignorirana.')
        self.flat_pairs = []
        for i in range(pair_count):
            self.flat_pairs.append((flat_jpg[2 * i], flat_jpg[2 * i + 1]))
        print('DataLoader (flat sequence): {} parova'.format(len(self.flat_pairs)))
    
    
    # test: get input images
    def get_data_clips(self, index, resize_height, resize_width):
        batch = []
        if self.flat_pairs is not None:
            left_path, right_path = self.flat_pairs[index]
            batch.append(np_load_frame(left_path, resize_height, resize_width))
            batch.append(np_load_frame(right_path, resize_height, resize_width))
        else:
            data_info_list = list(self.datas.values())
            for i in range(0, 2):
                image = np_load_frame(data_info_list[i]['frame'][index], resize_height, resize_width)
                batch.append(image)
        return np.concatenate(batch, axis=2)
    
    # test: get size
    def get_size_clips(self, index):
        batch = []
        if self.flat_pairs is not None:
            left_path, _ = self.flat_pairs[index]
            size = np_load_size(left_path)
        else:
            data_info_list = list(self.datas.values())
            size = np_load_size(data_info_list[0]['frame'][index])
        return size




def load(saver, sess, ckpt_path):
    #ckpt_path = 'checkpoints/stitch_rgb__lp_1.0_adv_0.0_gdl_0.0_flow_0.0/model.ckpt-600000'
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')




