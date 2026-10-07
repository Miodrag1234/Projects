from __future__ import print_function, division

import argparse
import csv
import json
import logging
import math
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from core.igev_stereo import IGEVStereo
from evaluate_stereo import *
import core.stereo_datasets as datasets

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


ALT_FEATURE_BACKBONES = (
    'mobilenetv3', 'mobilenetv3_large', 'mobilenetv3_small',
    'mobilenetv4', 'mobilenetv4_small', 'mobilenetv4_medium', 'mobilenetv4_large',
    'efficientnet_b0', 'efficientnet_lite0', 'efficientnet_lite',
    'resnet18', 'resnet50',
    'densenet', 'densenet121', 'densenet169',
)


def apply_backbone_training_defaults(args):
    """Stability defaults when feature encoder is not the original MobileNetV2 IGEV."""
    if args.feature_backbone not in ALT_FEATURE_BACKBONES:
        return
    if args.mixed_precision:
        logging.warning(
            f"Disabling --mixed_precision for {args.feature_backbone} (numerical stability)."
        )
        args.mixed_precision = False
    if args.lr == 0.0002:
        args.lr = 0.0001
        logging.info(f"{args.feature_backbone}: using lr=1e-4 (override default 2e-4).")


def load_checkpoint_compat(model, ckpt_path, feature_backbone):
    """Load IGEV checkpoint; skip feature.* when using an alternate backbone."""
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model_state = model.state_dict()
    if not any(k.startswith('module.') for k in checkpoint) and any(
        k.startswith('module.') for k in model_state
    ):
        checkpoint = {f'module.{k}': v for k, v in checkpoint.items()}
    to_load = {}
    skipped_feature = 0
    skipped_shape = 0
    for k, v in checkpoint.items():
        if k not in model_state:
            continue
        if feature_backbone != 'mobilenetv2' and '.feature.' in k:
            skipped_feature += 1
            continue
        if model_state[k].shape != v.shape:
            skipped_shape += 1
            continue
        to_load[k] = v
    model.load_state_dict(to_load, strict=False)
    logging.info(
        f"Loaded {len(to_load)}/{len(model_state)} tensors from {ckpt_path} "
        f"(skipped feature={skipped_feature}, shape={skipped_shape})"
    )


def append_validation_metrics(logdir, step, metrics, checkpoint=None, tag='val'):
    """Append EPE/D1 (and other validate_* keys) to JSON + CSV under logdir."""
    logdir = Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)

    record = {
        'step': int(step),
        'tag': tag,
        'checkpoint': str(checkpoint) if checkpoint else None,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in metrics.items()},
    }

    jsonl_path = logdir / 'validation_metrics.jsonl'
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')

    json_path = logdir / 'validation_metrics.json'
    history = []
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    history.append(record)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    flat = {'step': record['step'], 'tag': tag, 'checkpoint': record['checkpoint'] or ''}
    for k, v in record['metrics'].items():
        flat[k.replace('-', '_')] = v

    csv_path = logdir / 'validation_metrics.csv'
    rows = []
    fieldnames = list(flat.keys())
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            for row in reader:
                rows.append(row)
        for k in flat.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    rows.append({k: flat.get(k, '') for k in fieldnames})
    for k in flat.keys():
        if k not in fieldnames:
            fieldnames.append(k)
        rows[-1][k] = flat.get(k, '')

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    summary = ', '.join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record['metrics'].items())
    logging.info(f"Saved validation metrics (step {step}): {summary}")
    logging.info(f"  -> {jsonl_path.name}, {json_path.name}, {csv_path.name}")


def run_periodic_validation(model, args, step, use_mixed_precision):
    """Run dataset-specific validation; return merged metrics dict."""
    metrics = {}
    mixed = use_mixed_precision
    first_ds = args.train_datasets[0] if args.train_datasets else ''

    if 'sceneflow' in args.train_datasets:
        metrics.update(validate_sceneflow(model, iters=args.valid_iters, mixed_prec=mixed))

    if 'kitti' in first_ds:
        metrics.update(validate_kitti(model, iters=args.valid_iters, year=2012, root=args.kitti_root, mixed_prec=mixed))
        metrics.update(validate_kitti(model, iters=args.valid_iters, year=2015, root=args.kitti_root, mixed_prec=mixed))

    if first_ds == 'eth3d_finetune':
        metrics.update(validate_eth3d(model, iters=args.valid_iters, mixed_prec=mixed))

    if first_ds == 'middlebury_finetune':
        metrics.update(validate_middlebury(model, iters=args.valid_iters, mixed_prec=mixed))

    if first_ds == 'synthetic_train':
        synthetic_eval_root = getattr(args, 'synt_val_root', None) or getattr(args, 'synt_root', None)
        if synthetic_eval_root:
            metrics.update(validate_synthetic(
                model,
                iters=args.valid_iters,
                mixed_prec=mixed,
                root=synthetic_eval_root,
                disp_subdir=args.synt_disp_subdir,
                swap_lr=getattr(args, 'synt_swap_lr', False),
                disp_sign=getattr(args, 'synt_disp_sign', 1.0),
            ))
        else:
            logging.warning("synthetic_train: no --synt_root / --synt_val_root; skipping validation")

    return metrics


def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, loss_gamma=0.9, max_disp=192):
    """ Loss function defined over sequence of disp predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1

    disp_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(disp_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    # Cache the bool mask once (was being recomputed per loss term -> extra ops)
    valid_mask = valid.bool()
    if valid_mask.sum() == 0:
        metrics = {'epe': 0.0, '1px': 0.0, '3px': 0.0, '5px': 0.0}
        return None, metrics

    disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[valid_mask], disp_gt[valid_mask], size_average=True)
    for i in range(n_predictions):
        # We adjust the loss_gamma so it is consistent for any number of Selective-IGEV iterations
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid_mask].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    return disp_loss, metrics

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler

class Logger:
    SUM_FREQ = 100
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=args.logdir)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=args.logdir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=args.logdir)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def train(args):

    model = nn.DataParallel(IGEVStereo(args))
    print("Parameter Count: %d" % count_parameters(model))

    train_loader = datasets.fetch_dataloader(args)
    logging.info(f"Feature backbone: {args.feature_backbone}")
    apply_backbone_training_defaults(args)
    if args.feature_backbone in ('resnet18', 'resnet50', 'densenet121', 'densenet169') \
            and args.freeze_resnet_backbone:
        logging.info(
            f"{args.feature_backbone}: ImageNet backbone frozen; "
            f"training adapters + rest of IGEV."
        )
    if args.feature_backbone in ('mobilenetv3', 'mobilenetv3_large', 'mobilenetv3_small'):
        logging.info(
            "mobilenetv3: use --restore_ckpt with a pretrained IGEV .pth to load GRU/cost "
            "(feature weights are skipped; V3 trains from ImageNet + adapters)."
        )
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    logger = Logger(model, scheduler)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        if args.feature_backbone == 'mobilenetv2':
            checkpoint = torch.load(args.restore_ckpt)
            model.load_state_dict(checkpoint, strict=True)
        else:
            load_checkpoint_compat(model, args.restore_ckpt, args.feature_backbone)
        logging.info("Done loading checkpoint")
    model.cuda()
    model.train()
    model.module.freeze_bn() # We keep BatchNorm frozen

    validation_frequency = args.validation_frequency

    scaler = GradScaler(enabled=args.mixed_precision)
    use_mixed_precision = args.corr_implementation.endswith("_cuda")

    should_keep_training = True
    global_batch_num = 0
    last_validation_step = -1
    while should_keep_training:

        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2, disp_gt, valid = [x.cuda() for x in data_blob]

            assert model.training
            disp_init_pred, disp_preds = model(image1, image2, iters=args.train_iters)
            assert model.training

            loss, metrics = sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, max_disp=args.max_disp)
            if loss is None:
                if math.isnan(metrics.get('epe', 0.0)):
                    logging.warning(f"Skipping batch {i_batch}: non-finite predictions (try lower --lr, no --mixed_precision)")
                logger.push(metrics)
                optimizer.zero_grad(set_to_none=True)
                continue

            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_clip = 0.5 if args.feature_backbone in ALT_FEATURE_BACKBONES else 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            logger.push(metrics)

            if validation_frequency > 0 and total_steps % validation_frequency == validation_frequency - 1:
                val_step = total_steps + 1
                save_path = Path(args.logdir + '/%d_%s.pth' % (val_step, args.name))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)

                val_metrics = run_periodic_validation(model.module, args, val_step, use_mixed_precision)
                if val_metrics:
                    last_validation_step = val_step
                    append_validation_metrics(args.logdir, val_step, val_metrics, checkpoint=save_path, tag='val')
                    for key in val_metrics:
                        logger.write_dict({key: val_metrics[key]})
                    parts = []
                    for k, v in sorted(val_metrics.items()):
                        if 'epe' in k.lower():
                            parts.append(f"{k}={v:.4f}")
                        elif 'd1' in k.lower():
                            parts.append(f"{k}={v:.2f}%")
                    if parts:
                        print(f"[step {val_step}] " + ", ".join(parts))

                model.train()
                model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

        if len(train_loader) >= 10000:
            save_path = Path(args.logdir + '/%d_epoch_%s.pth.gz' % (total_steps + 1, args.name))
            logging.info(f"Saving file {save_path}")
            torch.save(model.state_dict(), save_path)

    # Final validation if training ended without a periodic val on the last checkpoint
    final_step = total_steps
    PATH = args.logdir + '/%s.pth' % args.name
    if validation_frequency > 0 and final_step != last_validation_step:
        logging.info("Running final validation...")
        final_metrics = run_periodic_validation(model.module, args, final_step, use_mixed_precision)
        if final_metrics:
            append_validation_metrics(args.logdir, final_step, final_metrics, checkpoint=PATH, tag='final')
            for k, v in sorted(final_metrics.items()):
                if 'epe' in k.lower():
                    print(f"FINAL {k}: {v:.4f}")
                elif 'd1' in k.lower():
                    print(f"FINAL {k}: {v:.2f}%")

    print("FINISHED TRAINING")
    logger.close()
    torch.save(model.state_dict(), PATH)
    logging.info(f"Saved final weights: {PATH}")
    print(f"Saved final weights: {PATH}")

    return PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='igev-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', default=None, help="load the weights from a specific checkpoint")
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float16', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--logdir', default=None, help='the directory to save logs and checkpoints')
    parser.add_argument('--kitti_root', default=None, help='path to KITTI root (folder containing 2012/ and 2015/)')
    parser.add_argument('--synt_root', default=None, help='path to synthetic dataset root (folder containing sceneXX/)')
    parser.add_argument('--synt_val_root', default=None, help='path to synthetic validation root (if set, final eval uses this instead of --synt_root)')
    parser.add_argument('--synt_disp_subdir', default='dispgt', choices=['dispgt', 'disp'], help='disparity subfolder in each synthetic scene')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 736], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=22, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')
    parser.add_argument('--validation_frequency', type=int, default=10000,
                        help='run validation every N steps (0 = only final eval)')

    # Architecure choices
    parser.add_argument(
        '--feature_backbone', default='mobilenetv2',
        choices=[
            'mobilenetv2',
            'mobilenetv3', 'mobilenetv3_large', 'mobilenetv3_small',
            'mobilenetv4', 'mobilenetv4_small', 'mobilenetv4_medium', 'mobilenetv4_large',
            'efficientnet_b0', 'efficientnet_lite0',
            'resnet18', 'resnet50',
            'densenet121', 'densenet169',
        ],
        help='feature encoder for matching (mobilenetv2 default; alt backbones need partial-load ckpt)',
    )
    parser.add_argument('--unfreeze_resnet_backbone', action='store_true',
                        help='also fine-tune ResNet18 weights (default: backbone frozen, only adapters train)')
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()

    torch.manual_seed(666)
    np.random.seed(666)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path(args.logdir).mkdir(exist_ok=True, parents=True)

    args.freeze_resnet_backbone = not args.unfreeze_resnet_backbone
    train(args)
