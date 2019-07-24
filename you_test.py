from darknet import Darknet
import dataset
from utils import *
from torchvision import datasets, transforms


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

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

    for batch_idx, (data, targets) in enumerate(valid_loader):
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
            prediction = boxes_to_prediction(boxes, width, height)
            get_prediction_metrics(prediction, targets)

