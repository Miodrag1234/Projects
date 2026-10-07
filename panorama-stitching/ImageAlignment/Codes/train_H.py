import tensorflow as tf
import os

from models import H_estimator, disjoint_augment_image_pair
from loss_functions import intensity_loss
from utils import load, save, DataLoader
import constant
import numpy as np


os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU

train_folder = constant.TRAIN_FOLDER
test_folder = constant.TEST_FOLDER

batch_size = constant.TRAIN_BATCH_SIZE
iterations = constant.ITERATIONS

height, width = 128, 128

summary_dir = constant.SUMMARY_DIR
snapshot_dir = constant.SNAPSHOT_DIR

# Ako je True, treniramo SAMO Net4 (feature_extract + Net1-Net3 su zamrznuti).
# Korisno za pravilno doucenje samo nove Net4 grane bez kvarenja vec naucenih Net1-Net3 weights.
FREEZE_BACKBONE = True
# Initial learning rate (vidi g_lrate ispod). Kad treniramo samo Net4, mozemo dignuti.
INITIAL_LR = 0.0005 if FREEZE_BACKBONE else 0.0001


# define dataset
with tf.name_scope('dataset'):
    ##########training###############
    ###input###
    train_data_loader = DataLoader(train_folder)
    train_data_dataset = train_data_loader(batch_size=batch_size)
    train_data_it = train_data_dataset.make_one_shot_iterator()
    (train_input_tensor, train_size_tensor) = train_data_it.get_next()
    train_input_tensor.set_shape([batch_size, height, width, 3*2])
    train_size_tensor.set_shape([batch_size, 2, 1])
    train_inputs = train_input_tensor
    train_size = train_size_tensor
    print('train inputs = {}'.format(train_inputs))
    

#only training dataset augment
with tf.name_scope('disjoint_augment'):
    train_inputs_aug = disjoint_augment_image_pair(train_inputs)



# define training generator function
with tf.variable_scope('generator', reuse=None):
    print('training = {}'.format(tf.get_variable_scope().name))
    (train_net1_f, train_net2_f, train_net3_f, train_net4_f,
     train_warp2_H1, train_warp2_H2, train_warp2_H3, train_warp2_H4,
     train_one_warp_H1, train_one_warp_H2, train_one_warp_H3, train_one_warp_H4) = H_estimator(train_inputs_aug, train_inputs, True)
   
   

with tf.name_scope('loss'):
    lam_lp = 1
    loss1 = intensity_loss(gen_frames=train_warp2_H1, gt_frames=train_inputs[...,0:3]*train_one_warp_H1, l_num=1)
    loss2 = intensity_loss(gen_frames=train_warp2_H2, gt_frames=train_inputs[...,0:3]*train_one_warp_H2, l_num=1)
    loss3 = intensity_loss(gen_frames=train_warp2_H3, gt_frames=train_inputs[...,0:3]*train_one_warp_H3, l_num=1)
    loss4 = intensity_loss(gen_frames=train_warp2_H4, gt_frames=train_inputs[...,0:3]*train_one_warp_H4, l_num=1)
    # Net4 dobiva najmanju tezinu jer radi finu korekciju nad H3 izlazom
    lp_loss = 16. * loss1 + 4. * loss2 + 1. * loss3 + 0.25 * loss4




with tf.name_scope('training'):
    g_loss = tf.add_n([lp_loss * lam_lp], name='g_loss')

    g_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='g_step')
    g_lrate = tf.train.exponential_decay(INITIAL_LR, g_step, decay_steps=50000/4, decay_rate=0.96)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=g_lrate, name='g_optimizer')
    g_vars_all = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    if FREEZE_BACKBONE:
        # Treniramo samo Net4 (i NE diramo feature_extract / Net1 / Net2 / Net3)
        g_vars = [v for v in g_vars_all if 'Reggression_Net4' in v.name]
        print('FREEZE_BACKBONE=True -> treniram samo Net4 ({} varijabli od {}).'
              .format(len(g_vars), len(g_vars_all)))
        for v in g_vars:
            print('  trainable:', v.name, v.shape.as_list())
    else:
        g_vars = g_vars_all
        print('Treniram sve generator varijable ({}).'.format(len(g_vars)))

    grads = g_optimizer.compute_gradients(g_loss, var_list=g_vars)
    for i, (g, v) in enumerate(grads):
      if g is not None:
        grads[i] = (tf.clip_by_norm(g, 3), v)  # clip gradients
    g_train_op = g_optimizer.apply_gradients(grads, global_step=g_step, name='g_train_op')
    

# add all to summaries
tf.summary.scalar(tensor=g_loss, name='g_loss')
tf.summary.scalar(tensor=loss1, name='loss1')
tf.summary.scalar(tensor=loss2, name='loss2')
tf.summary.scalar(tensor=loss3, name='loss3')
tf.summary.scalar(tensor=loss4, name='loss4')

tf.summary.image(tensor=train_inputs[...,0:3], name='train_inpu1')
tf.summary.image(tensor=train_inputs[...,3:6], name='train_inpu2')
tf.summary.image(tensor=train_warp2_H3, name='train_warp2_H3')
tf.summary.image(tensor=train_warp2_H4, name='train_warp2_H4')

summary_op = tf.summary.merge_all()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    # summaries
    summary_writer = tf.summary.FileWriter(summary_dir, graph=sess.graph)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)
    print("snapshot_dir")
    print(snapshot_dir)

    def permissive_load(sess, ckpt_path):
        """Ucitava samo varijable koje postoje u checkpointu i imaju isti shape.
           Nove varijable (npr. Reggression_Net4) ostaju random-inicijalizirane."""
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
        print('Permissive load: ucitavam {} varijabli, preskocim {} (nove ili promijenjenog shape-a).'
              .format(len(vars_to_load), len(skipped)))
        if skipped:
            print('  Primjer preskocenih:', skipped[:5])
        loader_p = tf.train.Saver(var_list=vars_to_load)
        loader_p.restore(sess, ckpt_path)
        print('Permissive load done.')

    if os.path.isdir(snapshot_dir):
        ckpt = tf.train.get_checkpoint_state(snapshot_dir)
        if ckpt and ckpt.model_checkpoint_path:
            permissive_load(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found.')
    else:
        permissive_load(sess, snapshot_dir)

    _step, _loss, _summaries = 0, None, None

    print("============starting training===========")
    import time, sys
    t_start = time.time()
    t_last = t_start
    last_step = 0
    while _step < iterations:
        try:
            _, _g_lr, _step, _lp_loss, _g_loss, _summaries = sess.run([g_train_op, g_lrate, g_step, lp_loss, g_loss, summary_op])

            now = time.time()
            dt = max(now - t_last, 1e-6)
            steps_done = _step - last_step
            it_per_sec = steps_done / dt if steps_done > 0 else 0.0
            t_last = now
            last_step = _step

            remaining = iterations - _step
            eta_sec = remaining / it_per_sec if it_per_sec > 0 else 0
            eta_h = int(eta_sec // 3600)
            eta_m = int((eta_sec % 3600) // 60)

            bar_w = 30
            frac = min(_step / float(iterations), 1.0)
            filled = int(bar_w * frac)
            bar = '=' * filled + '-' * (bar_w - filled)

            sys.stdout.write(
                "\r[{}] {}/{} ({:5.2f}%) | loss {:.4f} | lr {:.6f} | {:.2f} it/s | ETA {}h {:02d}m   "
                .format(bar, _step, iterations, frac * 100.0, _g_loss, _g_lr, it_per_sec, eta_h, eta_m)
            )
            sys.stdout.flush()

            if _step % 1000 == 0:
                summary_writer.add_summary(_summaries, global_step=_step)

            if _step % 5000 == 0 and _step > 0:
                print()
                save(saver, sess, snapshot_dir, _step)

        except tf.errors.OutOfRangeError:
            print('\nFinish successfully!')
            save(saver, sess, snapshot_dir, _step)
            break

    print('\nTraining loop finished at step {}. Spremam finalni checkpoint...'.format(_step))
    save(saver, sess, snapshot_dir, _step)
    print('Finalni checkpoint spremljen u {}'.format(snapshot_dir))
