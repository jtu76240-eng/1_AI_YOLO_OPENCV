import cv2
import numpy as np

from picamera2 import Picamera2

from hailo_platform import (
    HEF, Device, VDevice,
    InputVStreamParams, OutputVStreamParams,
    FormatType, HailoStreamInterface,
    InferVStreams, ConfigureParams
)

# 네가 학습한 7개 클래스 이름
CLASS_NAMES = [
    "H-beam",
    "coil",
    "deer",
    "human",
    "pallet",
    "rubber_cone",
    "tire",
]

# 박스 그릴 때 최소 신뢰도 / 최대 박스 개수
CONF_THRESHOLD = 0.3      # 0.2~0.5 사이에서 튜닝
MAX_BOXES = 30            # 프레임당 최대 박스 수

COLORS = np.random.uniform(0, 255, size=(len(CLASS_NAMES), 3))


def draw_bboxes(image, bboxes, confidences, class_ids):
    """프레임 위에 박스를 그리고 라벨/신뢰도 표시"""
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        cls_id = int(class_ids[i])
        conf = float(confidences[i])

        if 0 <= cls_id < len(CLASS_NAMES):
            label = f"{CLASS_NAMES[cls_id]}: {conf:.2f}"
        else:
            label = f"id{cls_id}: {conf:.2f}"

        color = COLORS[cls_id % len(COLORS)]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def main():
    hef_path = "/home/aicamera2/yolo_final/best.hef"

    print(f"[INFO] Loading HEF: {hef_path}")
    hef = HEF(hef_path)

    # Hailo 디바이스 스캔 및 VDevice 생성
    devices = Device.scan()
    if not devices:
        print("[ERROR] No Hailo devices found.")
        return

    target = VDevice(device_ids=devices)

    # 네트워크 그룹 설정 (Raspberry Pi 5 HAT는 PCIe)
    configure_params = ConfigureParams.create_from_hef(
        hef,
        interface=HailoStreamInterface.PCIe
    )
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    # 입력/출력 vstream 정보
    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_info = hef.get_output_vstream_infos()[0]

    print(f"[INFO] Input vstream:  name={input_vstream_info.name}, shape={input_vstream_info.shape}")
    print(f"[INFO] Output vstream: name={output_vstream_info.name}, shape={output_vstream_info.shape}")

    # input shape에서 H, W 추출 (지금은 (832, 832, 3))
    in_shape = input_vstream_info.shape
    if len(in_shape) == 3:
        in_h, in_w = in_shape[0], in_shape[1]
    elif len(in_shape) == 4:
        # (N, H, W, C) 가정
        in_h, in_w = in_shape[1], in_shape[2]
    else:
        print(f"[ERROR] Unexpected input shape: {in_shape}")
        return

    # float32로 주고받는 설정 (양자화 해제)
    input_vstreams_params = InputVStreamParams.make_from_network_group(
        network_group,
        quantized=False,
        format_type=FormatType.FLOAT32,
    )
    output_vstreams_params = OutputVStreamParams.make_from_network_group(
        network_group,
        quantized=False,
        format_type=FormatType.FLOAT32,
    )

    # Picamera2 설정 (RGB888, 832x832)
    picam2 = Picamera2()
    cam_config = picam2.create_preview_configuration(
        main={
            "size": (in_w, in_h),
            "format": "RGB888",
        }
    )
    picam2.configure(cam_config)
    picam2.start()

    print("[INFO] Camera started. Press 'q' to quit.")

    # NMS가 HEF 안에 포함되어 있다고 가정하고 tf_nms_format=True 사용
    with InferVStreams(
        network_group,
        input_vstreams_params,
        output_vstreams_params,
        tf_nms_format=True,
    ) as infer_pipeline:
        with network_group.activate(network_group_params):
            debug_printed = False
            try:
                while True:
                    # Picamera2에서 RGB 프레임 가져오기
                    frame_rgb = picam2.capture_array()

                    # 색: 일단 RGB 그대로 사용 (필요하면 BGR로 바꿔서 비교해도 됨)
                    # frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    frame = frame_rgb.copy()

                    h, w = frame.shape[:2]

                    # 모델 입력 크기에 맞춰 리사이즈 (이미 832x832면 영향 거의 없음)
                    resized = cv2.resize(frame, (in_w, in_h))

                    # 정규화: float32 / 255.0
                    input_tensor = resized.astype(np.float32) / 255.0

                    # 배치 차원 추가
                    input_data = {
                        input_vstream_info.name: np.expand_dims(input_tensor, axis=0)
                    }

                    # 추론
                    output_data = infer_pipeline.infer(input_data)

                    boxes = []
                    confidences = []
                    class_ids = []

                    # tf_nms_format=True → (num_classes, 5, num_dets)
                    # 여기서는 [ymin, xmin, ymax, xmax, score] 가정
                    for key, tensor in output_data.items():
                        arr = tensor[0]
                        if arr.ndim != 3:
                            if not debug_printed:
                                print(f"[WARN] Unexpected output ndim={arr.ndim}, shape={arr.shape}")
                            continue

                        num_classes, bbox_params, num_dets = arr.shape
                        if not debug_printed:
                            print(f"[DEBUG] arr.shape = {arr.shape} (num_classes, bbox_params, num_dets)")
                        if bbox_params < 5:
                            if not debug_printed:
                                print(f"[WARN] bbox_params < 5, shape={arr.shape}")
                            continue

                        # 디버그: 한 번만 통계 찍기
                        if not debug_printed:
                            conf_map = arr[:, 4, :]
                            print(f"[DEBUG] conf min={conf_map.min():.6f}, max={conf_map.max():.6f}")
                            sample = arr[0, :, 0]
                            print(f"[DEBUG] sample bbox (class 0, det 0) = {sample}")
                            debug_printed = True

                        for cls_id in range(min(num_classes, len(CLASS_NAMES))):
                            for det_id in range(num_dets):
                                bbox = arr[cls_id, :, det_id]

                                conf = float(bbox[4])
                                if conf < CONF_THRESHOLD:
                                    continue

                                # ★ 여기서부터가 핵심 수정 부분 ★
                                # TF 스타일: [ymin, xmin, ymax, xmax]
                                y1_norm = bbox[0]
                                x1_norm = bbox[1]
                                y2_norm = bbox[2]
                                x2_norm = bbox[3]

                                x1 = int(x1_norm * w)
                                y1 = int(y1_norm * h)
                                x2 = int(x2_norm * w)
                                y2 = int(y2_norm * h)

                                if x2 <= x1 or y2 <= y1:
                                    continue

                                boxes.append([x1, y1, x2, y2])
                                confidences.append(conf)
                                class_ids.append(cls_id)

                    # 너무 많으면 상위 conf만 남기기
                    if boxes:
                        boxes = np.array(boxes)
                        confidences = np.array(confidences)
                        class_ids = np.array(class_ids)

                        order = np.argsort(-confidences)
                        if len(order) > MAX_BOXES:
                            order = order[:MAX_BOXES]

                        boxes = boxes[order].tolist()
                        confidences = confidences[order].tolist()
                        class_ids = class_ids[order].tolist()

                        draw_bboxes(frame, boxes, confidences, class_ids)
                    else:
                        # 감지 없을 때 텍스트 출력
                        cv2.putText(
                            frame,
                            "No detections",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )

                    cv2.imshow("Hailo YOLO Detection (Picamera2)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            finally:
                picam2.stop()

    cv2.destroyAllWindows()
    print("[INFO] Finished.")


if __name__ == "__main__":
    main()
