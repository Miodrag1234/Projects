"""
Evaluacija robustnosti: razlicite rezolucije ulaznih slika (kao u UDIS radu).
Ispisuje postotak uspjesno spojenih parova i PSNR/SSIM po rezoluciji.

Pokreni dvaput za usporedbu:
  1) CKPT_MODE = 'udis_original'   -> njihov pretrenirani UDIS 1M
  2) CKPT_MODE = 'moj_model'        -> tvoj checkpoint (prilagodi putanju ispod)

Rezultati: evaluation_results_<CKPT_MODE>.txt
"""
import tensorflow as tf
import os
import numpy as np
import cv2

from models import H_estimator
from utils import DataLoader
import constant

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

# === KONFIGURACIJA ===
# 'udis_original'  = njihov pretrenirani model (1M, H3)
# 'moj_model'      = tvoj checkpoint (prilagodi snapshot_dir ispod)
# 'net4_only_h4'   = Net4-only fine-tune 1.2M (H4)
# 'finetune_h4'    = stari joint fine-tune 1.04M (H4)
CKPT_MODE = 'udis_original'

if CKPT_MODE == 'udis_original':
    snapshot_dir = './checkpoints_homo/model.ckpt-1000000'
    USE_NET4 = False
elif CKPT_MODE == 'moj_model':
    # PRILAGODI: tvoj najbolji ili zadnji checkpoint
    snapshot_dir = './checkpoints_homo3/model.ckpt-1200000'
    USE_NET4 = True
elif CKPT_MODE == 'net4_only_h4':
    snapshot_dir = './checkpoints_homo3/model.ckpt-1200000'
    USE_NET4 = True
elif CKPT_MODE == 'finetune_h4':
    snapshot_dir = './checkpoints_homo2/model.ckpt-1040000'
    USE_NET4 = True
else:
    raise ValueError('Nepoznat CKPT_MODE: ' + CKPT_MODE)

# Rezolucije (scale 1.0 = puni ulaz prije resize na 128x128 za mrezu)
RESOLUTION_SCALES = [1.0, 0.75, 0.5, 0.25]
OUTPUT_FILE = 'evaluation_results_{}.txt'.format(CKPT_MODE)

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU
test_folder = constant.TEST_FOLDER
batch_size = constant.TEST_BATCH_SIZE

with tf.name_scope('dataset'):
    test_inputs = tf.placeholder(shape=[batch_size, 128, 128, 3 * 2], dtype=tf.float32)

with tf.variable_scope('generator', reuse=None):
    (test_net1_f, test_net2_f, test_net3_f, test_net4_f,
     test_warp2_H1, test_warp2_H2, test_warp2_H3, test_warp2_H4,
     test_one_warp_H1, test_one_warp_H2, test_one_warp_H3, test_one_warp_H4) = H_estimator(
        test_inputs, test_inputs, False)

if USE_NET4:
    warp_tensor = test_warp2_H4
    mask_tensor = test_one_warp_H4
    output_label = 'H4 (Net4)'
else:
    warp_tensor = test_warp2_H3
    mask_tensor = test_one_warp_H3
    output_label = 'H3'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def permissive_load(sess, ckpt_path):
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
    loader = tf.train.Saver(var_list=vars_to_load)
    loader.restore(sess, ckpt_path)
    print('Permissive load done.')


def get_dataset_length(data_loader):
    if getattr(data_loader, 'flat_pairs', None) is not None:
        return len(data_loader.flat_pairs)
    return data_loader.datas['input1']['length']


def run_at_scale(sess, data_loader, scale):
    length = get_dataset_length(data_loader)
    h_in = max(32, int(128 * scale))
    w_in = max(32, int(128 * scale))
    psnr_list = []
    ssim_list = []
    success_count = 0

    for i in range(length):
        try:
            input_clip = np.expand_dims(data_loader.get_data_clips(i, h_in, w_in), axis=0)
            if h_in != 128 or w_in != 128:
                resized = cv2.resize(input_clip[0], (128, 128))
                input_clip = np.expand_dims(resized, axis=0)

            warp, warp_one = sess.run(
                [warp_tensor, mask_tensor],
                feed_dict={test_inputs: input_clip})

            warp = (warp + 1) * 127.5
            warp = warp[0]
            warp_one = warp_one[0]
            input1 = (input_clip[..., 0:3] + 1) * 127.5
            input1 = input1[0]

            psnr = compare_psnr(input1 * warp_one, warp * warp_one, data_range=255)
            ssim = _ssim(input1 * warp_one, warp * warp_one, data_range=255)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            success_count += 1
        except Exception:
            pass

    success_pct = 100.0 * success_count / length if length else 0
    mean_psnr = np.mean(psnr_list) if psnr_list else 0
    mean_ssim = np.mean(ssim_list) if ssim_list else 0
    return success_count, length, success_pct, mean_psnr, mean_ssim


with tf.Session(config=config) as sess:
    data_loader = DataLoader(test_folder)
    sess.run(tf.global_variables_initializer())

    print('Checkpoint:', snapshot_dir)
    print('Test folder:', test_folder)
    print('Izlaz:', output_label)
    print('CKPT_MODE:', CKPT_MODE)
    permissive_load(sess, snapshot_dir)

    lines = []
    lines.append('=== Evaluacija robustnosti (razlicite rezolucije) ===\n')
    lines.append('CKPT_MODE: {}\n'.format(CKPT_MODE))
    lines.append('Checkpoint: {}\n'.format(snapshot_dir))
    lines.append('Test folder: {}\n'.format(test_folder))
    lines.append('Izlaz: {}\n\n'.format(output_label))
    lines.append('Rezolucija (scale) | Uspjesno (N/ukupno) | Postotak | avg PSNR | avg SSIM\n')
    lines.append('-' * 75 + '\n')

    for scale in RESOLUTION_SCALES:
        ok, total, pct, mean_psnr, mean_ssim = run_at_scale(sess, data_loader, scale)
        res_str = 'scale={:.2f} ({:d}x{:d})'.format(
            scale, max(32, int(128 * scale)), max(32, int(128 * scale)))
        line = '{:25s} | {:6d}/{:6d} | {:6.2f}% | {:8.4f} | {:8.4f}\n'.format(
            res_str, ok, total, pct, mean_psnr, mean_ssim)
        lines.append(line)
        print(line.strip())

    lines.append('-' * 75 + '\n')
    text = ''.join(lines)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(text)
    print('\nRezultati spremljeni u:', os.path.abspath(OUTPUT_FILE))
