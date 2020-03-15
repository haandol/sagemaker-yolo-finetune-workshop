'''Finetune YOLOv3 with random shapes'''

# Built-Ins:
import os
import sys
import json
import time
import logging
import argparse
import warnings
import subprocess

# Install/Update GluonCV:
subprocess.call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'Cython'])
subprocess.call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'gluoncv', 'pycocotools'])

# External Dependencies:
import numpy as np
import mxnet as mx
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.dataloader import RandomTransformDataLoader
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.model_zoo import get_model
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from data import COCODetection
from mxnet import nd
from mxnet import gluon
from mxnet import autograd

logging.basicConfig()
logger = logging.getLogger('yolo3_darknet53')
logger.setLevel(logging.DEBUG)


def parse_args():
    hps = json.loads(os.environ["SM_HPS"])
    parser = argparse.ArgumentParser(description='Finetune YOLO networks with random input shape.')
    parser.add_argument('--network', type=str, default='yolo3_darknet53_coco',
                        help='Base network name which serves as feature extraction base.')
    parser.add_argument('--data-shape', type=int, default=320,
                        help='Input data shape for evaluation, use 320, 416, 608... ' +
                             'Training is with random shapes from (320 to 608).')
    parser.add_argument('--batch-size', type=int, default=4, help='Training mini-batch size')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=0, help='Number of data workers, you can use larger '
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'],
                        help='Number of GPUs to use in training.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Training epochs.')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                        'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='Optimizer used for training, default is sgd')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=40,
                        help='Logging mini-batch interval. Default is 40.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for testing, increase the number will reduce the '
                             'training time if test is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--save-interval', type=int, default=hps.get('save-interval', 10),
        help="Saving parameters epoch interval, best model will always be saved."
    )

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--images', type=str, default=os.environ['SM_CHANNEL_IMAGES'])
    args = parser.parse_args()
    return args


class GroundTruthDataset(gluon.data.Dataset):
    '''Custom Dataset to handle the GroundTruth json file'''
    def __init__(self, data_path, channel, image_path, field_name):
        '''
        Parameters
        ---------
        data_path: str, Path to the data folder, default 'data'
        field_name: str, The annotation task name that appears in your json
                    the parent node of `annotations` that holds bbs infos
        '''
        self.data_path = data_path
        self.image_path = image_path
        self.field_name = field_name
        self.image_info = []
        with open(os.path.join(data_path, '{}.manifest'.format(channel))) as f:
            lines = f.readlines()
            for line in lines:
                info = json.loads(line[:-1])
                if len(info[field_name]['annotations']):
                    self.image_info.append(info)

    def __getitem__(self, idx):
        '''
        Parameters
        ---------
        idx: int, index requested

        Returns
        -------
        image: nd.NDArray
            The image
        label: np.NDArray bounding box labels of the form [[x1,y1, x2, y2, class], ...]
        '''
        info = self.image_info[idx]
        source_ref = (
            info['source-ref'][5:].partition('/')[2]
            if (info['source-ref'][:5] == 's3://')
            else info['source-ref']
        )
        image = mx.image.imread(
            os.path.join(self.image_path, *source_ref.split('/'))
        )
        boxes = info[self.field_name]['annotations']
        label = []
        for box in boxes:
            label.append([
                box['left'], box['top'],
                box['left']+box['width'], box['top']+box['height'],
                0
            ])

        return image, np.array(label)

    def __len__(self):
        return len(self.image_info)


def get_dataset(args):
    train_dataset = COCODetection(args.train, 'train', args.images, 'labels')
    val_dataset = COCODetection(args.test, 'test', args.images, 'labels')

    val_metric = COCODetectionMetric(
        val_dataset, args.save_prefix + '_eval', cleanup=True,
        data_shape=(args.data_shape, args.data_shape)
    )

    args.num_samples = len(train_dataset)
    return train_dataset, val_dataset, val_metric


def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers, args):
    batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))  # stack image, all targets generated
    transform_fns = [
        YOLO3DefaultTrainTransform(x * 32, x * 32, net)
        for x in range(10, 20)
    ]
    train_loader = RandomTransformDataLoader(
        transform_fns, train_dataset, batch_size=batch_size, interval=10,
        last_batch='rollover', shuffle=True, batchify_fn=batchify_fn,
        num_workers=num_workers
    )

    width, height = data_shape, data_shape
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultValTransform(width, height)),
        batch_size, True, batchify_fn=val_batchify_fn, last_batch='rollover',
        num_workers=num_workers
    )
    return train_loader, val_loader


def save_params(net, best_loss, current_loss, epoch, save_interval, model_dir):
    if current_loss < best_loss[0]:
        logger.info(f'save model with current loss: {current_loss:2.6f}, previous best loss: {best_loss[0]:2.6f}')
        best_loss[0] = current_loss
        net.save_parameters(os.path.join(model_dir, f'model.params'))


def validate(net, val_data, ctx, eval_metric):
    """Test on test-dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    mx.nd.waitall()
    net.hybridize()

    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids)
    return eval_metric.get()


def train(net, train_data, val_data, eval_metric, ctx, args):
    net.collect_params().reset_ctx(ctx)
    if args.optimizer =='sgd':
        trainer = gluon.Trainer(
            net.collect_params(), args.optimizer,
            {'wd': args.wd, 'momentum': args.momentum, 'learning_rate': args.lr},
            kvstore='local'
        )
    elif args.optimizer =='adam':
        trainer = gluon.Trainer(
            net.collect_params(), args.optimizer,
            {'learning_rate': args.lr},
            kvstore='local'
        )
    else:
        trainer = gluon.Trainer(
            net.collect_params(), args.optimizer, kvstore='local'
        )

    # metrics
    sum_metrics = mx.metric.Loss('SumLoss')
    obj_metrics = mx.metric.Loss('ObjLoss')
    center_metrics = mx.metric.Loss('BoxCenterLoss')
    scale_metrics = mx.metric.Loss('BoxScaleLoss')
    cls_metrics = mx.metric.Loss('ClassLoss')

    # set up logger
    logger.info('Start training')
    best_loss = [float('inf')]
    for epoch in range(args.start_epoch, args.epochs):
        tic = time.time()
        btic = time.time()
        mx.nd.waitall()
        net.hybridize()
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            # objectness, center_targets, scale_targets, weights, class_targets
            fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0, even_split=False) for it in range(1, 6)]
            gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0, even_split=False)
            sum_losses = []
            obj_losses = []
            center_losses = []
            scale_losses = []
            cls_losses = []
            with autograd.record():
                for ix, x in enumerate(data):
                    obj_loss, center_loss, scale_loss, cls_loss = net(x, gt_boxes[ix], *[ft[ix] for ft in fixed_targets])
                    sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                    obj_losses.append(obj_loss)
                    center_losses.append(center_loss)
                    scale_losses.append(scale_loss)
                    cls_losses.append(cls_loss)
                autograd.backward(sum_losses)
            trainer.step(batch_size)

            sum_metrics.update(0, sum_losses)
            obj_metrics.update(0, obj_losses)
            center_metrics.update(0, center_losses)
            scale_metrics.update(0, scale_losses)
            cls_metrics.update(0, cls_losses)
            if args.log_interval and not (i + 1) % args.log_interval:
                name0, loss0 = sum_metrics.get()
                name1, loss1 = obj_metrics.get()
                name2, loss2 = center_metrics.get()
                name3, loss3 = scale_metrics.get()
                name4, loss4 = cls_metrics.get()
                logger.info('[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, trainer.learning_rate, batch_size/(time.time()-btic),
                    name0, loss0, name1, loss1, name2, loss2, name3, loss3, name4, loss4
                ))
            btic = time.time()

        name0, loss0 = sum_metrics.get()
        name1, loss1 = obj_metrics.get()
        name2, loss2 = center_metrics.get()
        name3, loss3 = scale_metrics.get()
        name4, loss4 = cls_metrics.get()
        logger.info('[Epoch {}] Train Loss: {} ;'.format(epoch, loss0))
        logger.info('[Epoch {}] Train Cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time()-tic), name1, loss1, name2, loss2, name3, loss3, name4, loss4
        ))

#         if not (epoch + 1) % args.val_interval:
#             map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
#             val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
#             logger.info('[Epoch {}] Test: {}'.format(epoch, val_msg))
        
        save_params(net, best_loss, float(loss0), epoch, args.save_interval, args.model_dir)


if __name__ == '__main__':
    from glob import glob
    print(glob('/opt/ml/input/data/images/*'))

    args = parse_args()

    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in range(args.num_gpus)]
    ctx = ctx if ctx else [mx.cpu()]

    # network
    net_name = args.network
    args.save_prefix += net_name

    net = get_model(net_name, pretrained=True)
    net.reset_class(classes=['person'], reuse_weights=['person'])

    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(args)
    train_data, val_data = get_dataloader(
        net, train_dataset, val_dataset,
        args.data_shape, args.batch_size, args.num_workers, args
    )

    # training
    train(net, train_data, val_data, eval_metric, ctx, args)