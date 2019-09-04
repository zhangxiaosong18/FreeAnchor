# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import datetime
import logging
import time
import os

import torch

from tqdm import tqdm

from maskrcnn_benchmark.utils.comm import is_main_process
from maskrcnn_benchmark.utils.comm import scatter_gather
from maskrcnn_benchmark.utils.comm import synchronize


from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou


def compute_on_dataset(model, data_loader, device):
    model.eval()
    with_results_dict = {}
    without_results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in tqdm(enumerate(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with_overlaps = []
        without_overlaps = []
        with torch.no_grad():
            with_outputs, without_outputs = model(images)
            for output, target in zip(with_outputs, targets):
                if len(target) == 0:
                    continue
                if len(output) == 0:
                    overlap = torch.zeros(len(target), device=cpu_device, dtype=torch.float)
                    with_overlaps.append(overlap)
                else:
                    overlap = boxlist_iou(output, target.to(device)).max(dim=0).values
                    with_overlaps.append(overlap.to(cpu_device))
            for output, target in zip(without_outputs, targets):
                if len(target) == 0:
                    continue
                if len(output) == 0:
                    overlap = torch.zeros(len(target), device=cpu_device, dtype=torch.float)
                    without_overlaps.append(overlap)
                else:
                    overlap = boxlist_iou(output, target.to(device)).max(dim=0).values
                    without_overlaps.append(overlap.to(cpu_device))
        with_results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, with_overlaps)}
        )
        without_results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, without_overlaps)}
        )
    return with_results_dict, without_results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = scatter_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.eval_IR")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def evaluate_iou_average_recall(with_overlaps, without_overlaps, output_folder):
    results = []

    with_overlaps = torch.cat(with_overlaps).numpy()
    without_overlaps = torch.cat(without_overlaps).numpy()

    iouThrs = np.arange(0.5, 0.95, 0.05)

    matched = (with_overlaps > iouThrs[:, None]).astype(np.int64)
    with_iou_recalls = matched.sum(1)

    matched = (without_overlaps > iouThrs[:, None]).astype(np.int64)
    without_iou_recalls = matched.sum(1)
    nms_iou_recalls = with_iou_recalls / without_iou_recalls.clip(min=1)
    torch.save([iouThrs, nms_iou_recalls], os.path.join(output_folder, 'iou_recalls.pth'))
    print('NMS Recall (NR) [0.5:0.9] = {:0.3f}'.format(float(nms_iou_recalls.mean())))
    for t, NR in zip(iouThrs[0::2], nms_iou_recalls[0::2]):
        results.append(NR)
        print('NMS Recall (NR) [IoU > {:0.2f}] = {:0.3f}'.format(t, float(NR)))
    return results


def inference(
    model,
    data_loader,
    iou_types=("bbox",),
    box_only=False,
    device="cuda",
    expected_results=(),
    expected_results_sigma_tol=4,
    without_nms=False,
    output_folder=None,
):

    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.deprecated.get_world_size()
        if torch.distributed.deprecated.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.eval_IR")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} images".format(len(dataset)))
    start_time = time.time()
    with_overlaps, without_overlaps = compute_on_dataset(model, data_loader, device)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    with_overlaps = _accumulate_predictions_from_multiple_gpus(with_overlaps)
    without_overlaps = _accumulate_predictions_from_multiple_gpus(without_overlaps)
    if not is_main_process():
        return

    logger.info("Evaluating IoU average Recall (IR)")
    results = evaluate_iou_average_recall(with_overlaps, without_overlaps, output_folder)
    logger.info(results)
