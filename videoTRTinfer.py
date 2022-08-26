#https://github.com/Linaom1214/TensorRT-For-YOLO-Series 

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2

engine_path = '/your/path/to/engine/yolov7-tiny-nms.trt'
class_names = ['none','mom', 'Miha']
imgsz=(640,640)
mean = None
std = None
end2end = "use end2end engine"
n_classes = 2
conf = 0.5

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
trt.init_libnvinfer_plugins(logger, '')  # initialize TensorRT plugins
with open(engine_path, "rb") as f:
    serialized_engine = f.read()
engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()
inputs, outputs, bindings = [], [], []
stream = cuda.Stream()
for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding))
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))
    print("bindings =", binding)
    if engine.binding_is_input(binding):
        inputs.append({'host': host_mem, 'device': device_mem})
        # print(inputs)
    else:
        outputs.append({'host': host_mem, 'device': device_mem})
        # print(outputs)

def infer(img):
    inputs[0]['host'] = np.ravel(img)
    # transfer data to the gpu
    for inp in inputs:
        cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
    # run inference
    context.execute_async_v2(
        bindings= bindings,
        stream_handle= stream.handle)
    # fetch outputs from gpu
    for out in outputs:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
    # synchronize stream
    stream.synchronize()

    data = [out['host'] for out in outputs]
    return data

def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    # if use yolox set
    # padded_img = padded_img[:, :, ::-1]
    # padded_img /= 255.0
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.667, 0.000, 1.000,
        0.929, 0.694, 0.125
    ]
).astype(np.float32).reshape(-1, 3)


def postprocess(predictions, ratio):
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    return dets


def get_fps(self):
    # warmup
    import time
    img = np.ones((1, 3, self.imgsz[0], self.imgsz[1]))
    img = np.ascontiguousarray(img, dtype=np.float32)
    for _ in range(20):
        _ = self.infer(img)
    t1 = time.perf_counter()
    _ = self.infer(img)
    print(1 / (time.perf_counter() - t1), 'FPS')


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.7, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.6, txt_color, thickness=1)

    return img
video_path = "/your/input/video/path"

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('/home/mih/Видео/results.avi', fourcc, fps, (width, height))

fps = 0
import time
t_tick = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    blob, ratio = preproc(frame, imgsz, mean, std)
    t1 = time.time()
    start_time = cv2.getTickCount()
    data = infer(blob)

    if end2end:
        num, final_boxes, final_scores, final_cls_inds = data
        final_boxes = np.reshape(final_boxes / ratio, (-1, 4))
        dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1),
                               np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)

    else:
        predictions = np.reshape(data, (1, -1, int(5 + n_classes)))[0]
        dets = postprocess(predictions, ratio)

    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:,
                                                    :4], dets[:, 4], dets[:, 5]
        frame = vis(frame, final_boxes, final_scores, final_cls_inds,
                    conf=conf, class_names= class_names)

        t_tick = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        fps = 1 / t_tick
        print("fps = ", fps)
        frame = cv2.putText(frame, "FPS:%d " % fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    # out.write(frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
out.release()
cap.release()
cv2.destroyAllWindows()

