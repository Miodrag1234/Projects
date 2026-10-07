import tensorflow as tf
import os
import numpy as np
import cv2 as cv

from models import H_estimator
from utils import DataLoader, load, save
import constant
# skimage: novije verzije imaju PSNR/SSIM u metrics, starije u measure
try:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim
    def _ssim(a, b, data_range=255):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        h, w = a.shape[0], a.shape[1]
        win_size = min(7, h, w)
        if win_size % 2 == 0:
            win_size = max(3, win_size - 1)
        win_size = max(3, win_size)
        try:
            return compare_ssim(a, b, data_range=data_range, channel_axis=-1, win_size=win_size)
        except (ValueError, TypeError):
            return compare_ssim(a, b, data_range=data_range, multichannel=True, win_size=win_size)
except ImportError:
    from skimage.measure import compare_psnr, compare_ssim
    def _ssim(a, b, data_range=255):
        return compare_ssim(a, b, data_range=data_range, multichannel=True)

slim = tf.contrib.slim

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU
test_folder = constant.TEST_FOLDER

# === KONFIGURACIJA EVALUACIJE ===
# CKPT_MODE odreduje koji checkpoint i koji izlaz koristimo:
#   'original_h3'     -> originalni 1M, H3 izlaz (baseline)
#   'finetune_h4'     -> stari joint fine-tune 1.04M, H4 izlaz
#   'finetune_h3'     -> stari joint fine-tune 1.04M, H3 izlaz (dijagnostika)
#   'net4_only_h4'    -> Net4-only fine-tune 1.2M, H4 izlaz (PREPORUCENO za novi model)
#   'net4_only_h3'    -> Net4-only fine-tune 1.2M, H3 izlaz (dijagnostika: Net1-Net3 ostali netaknuti?)
CKPT_MODE = 'net4_only_h4'

if CKPT_MODE == 'original_h3':
    snapshot_dir = './checkpoints_homo/model.ckpt-1000000'
    USE_NET4 = False
elif CKPT_MODE == 'finetune_h4':
    snapshot_dir = './checkpoints_homo2/model.ckpt-1040000'
    USE_NET4 = True
elif CKPT_MODE == 'finetune_h3':
    snapshot_dir = './checkpoints_homo2/model.ckpt-1040000'
    USE_NET4 = False
elif CKPT_MODE == 'net4_only_h4':
    snapshot_dir = './checkpoints_homo3/model.ckpt-1200000'
    USE_NET4 = True
elif CKPT_MODE == 'net4_only_h3':
    snapshot_dir = './checkpoints_homo3/model.ckpt-1200000'
    USE_NET4 = False
else:
    raise ValueError("Nepoznat CKPT_MODE: " + CKPT_MODE)

batch_size = constant.TEST_BATCH_SIZE

# define dataset
with tf.name_scope('dataset'):
    ##########testing###############
    
    test_inputs = tf.placeholder(shape=[batch_size, 128, 128, 3 * 2], dtype=tf.float32)
    print('test inputs = {}'.format(test_inputs))



with tf.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    (test_net1_f, test_net2_f, test_net3_f, test_net4_f,
     test_warp2_H1, test_warp2_H2, test_warp2_H3, test_warp2_H4,
     test_one_warp_H1, test_one_warp_H2, test_one_warp_H3, test_one_warp_H4) = H_estimator(test_inputs, test_inputs, False)
   


config = tf.ConfigProto()
config.gpu_options.allow_growth = True      
with tf.Session(config=config) as sess:
    # dataset
    data_loader = DataLoader(test_folder)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)

    def permissive_load(sess, ckpt_path):
        """Ucitava samo varijable koje postoje u checkpointu i imaju isti shape.
           Nove varijable (npr. Reggression_Net4 u starom 1M checkpointu)
           ostaju random-inicijalizirane."""
        reader = tf.train.NewCheckpointReader(ckpt_path)
        saved_shapes = reader.get_variable_to_shape_map()
        vars_to_load = []
        skipped = []
        for v in tf.global_variables():
            name = v.name.split(':')[0]
            if name in saved_shapes and v.shape.as_list() == saved_shapes[name]:
                vars_to_load.append(v)
            else:
                skipped.append(name)
        print('Permissive load: ucitavam {} varijabli, preskocim {}.'
              .format(len(vars_to_load), len(skipped)))
        if skipped:
            print('  Primjer preskocenih:', skipped[:5])
        loader_p = tf.train.Saver(var_list=vars_to_load)
        loader_p.restore(sess, ckpt_path)
        print('Permissive load done.')

    def inference_func(ckpt):
        print("============")
        print(ckpt)
        permissive_load(sess, ckpt)
        print("============")
        if getattr(data_loader, 'flat_pairs', None) is not None:
            length = len(data_loader.flat_pairs)
        else:
            length = data_loader.datas['input1']['length']
        psnr_list = []
        ssim_list = []

        # Biraj izlaz: H4 za novi model s Net4, H3 za originalni 1M (Net4 random-init -> garbage)
        if USE_NET4:
            warp_tensor, mask_tensor = test_warp2_H4, test_one_warp_H4
            print("Koristim H4 izlaz (Net4 fine-tuned model)")
        else:
            warp_tensor, mask_tensor = test_warp2_H3, test_one_warp_H3
            print("Koristim H3 izlaz (originalni 1M model, Net4 nije treniran)")

        for i in range(0, length):
            #load test data
            input_clip = np.expand_dims(data_loader.get_data_clips(i, 128, 128), axis=0)
            
            warp, warp_one = sess.run([warp_tensor, mask_tensor], feed_dict={test_inputs: input_clip})
            
            
            warp = (warp+1) * 127.5    
            warp = warp[0] 
            warp_one = warp_one[0]
            input1 = (input_clip[...,0:3]+1) * 127.5    
            input1 = input1[0]
            input2 = (input_clip[...,3:6]+1) * 127.5    
            input2 = input2[0]
            
            # PSNR/SSIM samo u podrucju preklapanja: warp_one = maska preklapanja (gdje je warp valjan)
            # Usporedba: referentna slika (input1) vs. poravnata druga slika (warp) u tom podrucju
            psnr = compare_psnr(input1*warp_one, warp*warp_one, data_range=255)
            ssim = _ssim(input1*warp_one, warp*warp_one, data_range=255)

            
            print('i = {} / {}, psnr = {:.6f}'.format( i+1, length, psnr))
            
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
            
        print("===================Results Analysis==================")
        n = len(psnr_list)
        print("Uspjesno spojeno: {} / {} ({:.1f}%)".format(n, length, 100.0 * n / length if length else 0))
        i30 = int(n * 0.30)
        i60 = int(n * 0.60)
        psnr_list.sort(reverse=True)
        psnr_list_30 = psnr_list[0:i30]
        psnr_list_60 = psnr_list[i30:i60]
        psnr_list_100 = psnr_list[i60:]
        print("top 30%", np.mean(psnr_list_30))
        print("top 30~60%", np.mean(psnr_list_60))
        print("top 60~100%", np.mean(psnr_list_100))
        print('average psnr:', np.mean(psnr_list))

        ssim_list.sort(reverse=True)
        ssim_list_30 = ssim_list[0:i30]
        ssim_list_60 = ssim_list[i30:i60]
        ssim_list_100 = ssim_list[i60:]
        print("top 30%", np.mean(ssim_list_30))
        print("top 30~60%", np.mean(ssim_list_60))
        print("top 60~100%", np.mean(ssim_list_100))
        print('average ssim:', np.mean(ssim_list))

    inference_func(snapshot_dir)
    

