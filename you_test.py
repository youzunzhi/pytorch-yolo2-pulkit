from darknet import Darknet
import dataset
import numpy as np
import torch
import tqdm
from utils import nms, get_region_boxes, get_image_size
from torchvision import datasets, transforms


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def rescale_target(targets_i, width, height):
    targets_i_list = []
    for t in range(50):
        if targets_i[t*5] == 0:
            break
        targets_i_list.append(targets_i[t*5:t*5+5])
    you_targets_i = torch.stack(targets_i_list)
    you_targets_i[:, 1:] = xywh2xyxy(you_targets_i[:, 1:])
    you_targets_i[:, 1] *= width
    you_targets_i[:, 2] *= height
    you_targets_i[:, 3] *= width
    you_targets_i[:, 4] *= height
    return you_targets_i

def boxes_to_prediction(boxes, width, height):
    prediction = torch.zeros((len(boxes), 7))
    for i, box in enumerate(boxes):
        prediction[i, 0] = (box[0] - box[2] / 2.0) * width
        prediction[i, 1] = (box[1] - box[3] / 2.0) * height
        prediction[i, 2] = (box[0] + box[2] / 2.0) * width
        prediction[i, 3] = (box[1] + box[3] / 2.0) * height
        prediction[i, 4] = box[4]
        prediction[i, 5] = box[5]
        prediction[i, 6] = box[6]

    return prediction

def get_prediction_metrics(prediction, targets):
    """

    :param prediction: [n_pred_box, 7(4coord+obj_conf+cls_conf+cls)]
    :param targets: [n_target_box, 5(label+4coord)]
    :return:
    """
    iou_thresh = 0.5

    pred_boxes = prediction[:, :4]
    pred_conf = prediction[:, 4]
    pred_labels = prediction[:, -1]

    true_positives = np.zeros(pred_boxes.shape[0])

    target_labels = targets[:, 0] if len(targets) else []
    if len(targets):
        detected_boxes = []
        target_boxes = targets[:, 1:]

        for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

            # If targets are all found then break
            if len(detected_boxes) == len(targets):
                break

            # Ignore if label is not one of the target labels
            if pred_label not in target_labels:
                continue

            iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
            if iou >= iou_thresh and box_index not in detected_boxes and pred_label == target_labels[box_index]:
                true_positives[pred_i] = 1
                detected_boxes += [box_index]
    return [true_positives, pred_conf, pred_labels]

def show_eval_result(metrics, labels):
    true_positives, pred_conf, pred_labels = [np.concatenate(x, 0) for x in list(zip(*metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_conf, pred_labels, labels)
    print(f"mAP: {AP.mean()}")

if __name__ == '__main__':
    import pydevd_pycharm

    pydevd_pycharm.settrace('172.26.3.54', port=12343, stdoutToServer=True, stderrToServer=True)

    datacfg = 'cfg/voc.data'
    cfgfile = 'cfg/yolov2-tiny-voc.cfg'
    weightfile = '../yolov2-tiny-bnn/weights/yolov2-tiny-voc.weights'
    options = read_data_cfg(datacfg)
    valid_images = options['valid']
    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]

    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    m.cuda()
    m.eval()

    valid_dataset = dataset.listDataset(valid_images, shape=(m.width, m.height),
                                        shuffle=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                        ]))
    valid_batchsize = 2
    assert (valid_batchsize > 1)

    kwargs = {'num_workers': 0, 'pin_memory': True}
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs)

    conf_thresh = 0.005
    nms_thresh = 0.45
    metrics = []
    labels = []

    for batch_idx, (data, targets) in enumerate(tqdm.tqdm(valid_loader, desc="Detecting objects")):
        data = data.cuda()
        data = Variable(data, volatile=True)
        output = m(data).data
        batch_boxes = get_region_boxes(output, conf_thresh, m.num_classes, m.anchors, m.num_anchors, 0, 1)
        for i in range(output.size(0)):
            targets_i = targets[i]
            boxes = batch_boxes[i]
            boxes = nms(boxes, nms_thresh)
            width, height = get_image_size(valid_files[i])
            targets_i = rescale_target(targets_i, width, height)
            labels += targets_i[:, 0].tolist()
            prediction = boxes_to_prediction(boxes, width, height)
            metrics += get_prediction_metrics(prediction, targets_i)

    show_eval_result(metrics, labels)




def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

