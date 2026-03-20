import cv2
import numpy as np
import time
import struct
import threading

from hailo_platform import (
    HEF, Device, VDevice,
    InputVStreamParams, OutputVStreamParams,
    FormatType, HailoStreamInterface,
    InferVStreams, ConfigureParams
)

# ==========================================
# 기본 설정
# ==========================================
IMG_WIDTH, IMG_HEIGHT = 832, 832
LANE_PROCESS_HZ = 1.0              # 차선 처리 주기
LANE_SEND_HZ   = 1.0               # UART로 lane 결과 보낼 주기
OBJ_SEND_HZ    = 20.0              # UART로 YOLO 결과 보낼 주기
CAPTURE_SLEEP  = 0.0               # 과열/부하 시 0.001~0.01

# [TUNE]
GAMMA_VALUE = 1.70
BOTTOM_MASK_HEIGHT = 0

# ==========================================
# UART 설정
# ==========================================
UART_DEVICE = "/dev/ttyAMA0"
UART_BAUD   = 9600

_seq = 0

# ==========================================
# YOLO(Hailo) 설정
# ==========================================
HAILO_HEF_PATH = "/home/aicamera2/yolo_final/best.hef"

# 학습한 7개 클래스 이름
CLASS_NAMES = [
    "H-beam",
    "coil",
    "deer",
    "human",
    "pallet",
    "rubber_cone",
    "tire",
]

# Hailo 박스 필터링 파라미터
HAILO_CONF_THRESHOLD = 0.3     # 0.2~0.5 사이에서 튜닝
HAILO_MAX_BOXES      = 5       # 프레임당 그릴 최대 박스 개수

# "객체는 최대 3개만" UART 전송
MAX_SEND_OBJECTS = 3

# ==========================================
# Lane 파라미터 (Detect only)
# ==========================================
# [TUNE] ROI=0.68, TOP=0.44, ZOOM=0.25
ROI_HEIGHT_RATIO_DETECT = 0.60
TRAPEZOID_TOP_MARGIN_DETECT = 0.44
BEV_ZOOM_FACTOR_DETECT = 0.27

# BGR 기준 노란선/흰선 임계값 (튜닝 버전)
YELLOW_BGR_LOWER = np.array([50, 100, 130])
YELLOW_BGR_UPPER = np.array([80, 255, 255])
WHITE_BGR_LOWER  = np.array([130, 130, 130])
WHITE_BGR_UPPER  = np.array([255, 255, 255])

# ---------------------------
# 해상도 의존 파라미터 
# ---------------------------
# [TUNE] WIN_MARGIN=150, MIN_PIX=118, N_WIN=15, LANE_W=210
WINDOW_MARGIN = 150
MIN_PIX = 118
N_WINDOWS = 15
LANE_WIDTH_PX = 210

WALL_Y_LIMIT   = 173
WALL_MIN_WIDTH = 62
WALL_MIN_AREA  = 1800

MIN_PEAK_DISTANCE = 156
MAX_CURVATURE     = 0.010

last_lane_status = "Undefined"
miss_count = 0

# ==========================================
# 공유 데이터
# ==========================================
_latest_frame = None
_frame_lock = threading.Lock()

_latest_lane_final = None   # BEV + 차선 결과 (BGR)
_lane_lock = threading.Lock()

_latest_yolo_frame = None   # YOLO 표시용
_yolo_lock = threading.Lock()

_latest_lane_tx = None      # {"status_id": int, "status_str": str}
_lane_tx_lock = threading.Lock()

_latest_yolo_tx = None      # [{"id":int,"name":str,"conf":float,"xc":float,"yc":float}, ...]
_yolo_tx_lock = threading.Lock()

_stop_event = threading.Event()


def set_latest_frame(frame):
    global _latest_frame
    with _frame_lock:
        _latest_frame = frame


def get_latest_frame_copy():
    with _frame_lock:
        if _latest_frame is None:
            return None
        return _latest_frame.copy()


def set_lane_final(img_bgr):
    global _latest_lane_final
    with _lane_lock:
        _latest_lane_final = img_bgr


def get_lane_final():
    with _lane_lock:
        return _latest_lane_final


def set_yolo_frame(img):
    global _latest_yolo_frame
    with _yolo_lock:
        _latest_yolo_frame = img


def get_yolo_frame():
    with _yolo_lock:
        return _latest_yolo_frame


def set_lane_tx(tx):
    global _latest_lane_tx
    with _lane_tx_lock:
        _latest_lane_tx = tx


def get_lane_tx():
    with _lane_tx_lock:
        return _latest_lane_tx


def set_yolo_tx(objs):
    global _latest_yolo_tx
    with _yolo_tx_lock:
        _latest_yolo_tx = objs


def get_yolo_tx():
    with _yolo_tx_lock:
        return _latest_yolo_tx


# ==========================================
# Lane 유틸
# ==========================================
def status_to_id(status_str: str) -> int:
    s = status_str.replace("(Mem)", "").replace("(Est)", "").strip()
    if s.startswith("Lane 1"):        return 1
    if s.startswith("Between 1-2"):   return 2
    if s.startswith("Lane 2"):        return 3
    if s.startswith("Between 2-3"):   return 4
    if s.startswith("Lane 3"):        return 5
    return 0


_gamma_cache = {"val": None, "lut": None}


def apply_gamma(image_bgr, gamma=1.0):
    global _gamma_cache
    if _gamma_cache["val"] != gamma:
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        _gamma_cache["val"] = gamma
        _gamma_cache["lut"] = table
    return cv2.LUT(image_bgr, _gamma_cache["lut"])


def get_roi_points(w, h, roi_height_ratio, top_margin):
    roi_h = int(h * roi_height_ratio)
    roi_start_y = h - roi_h
    return np.float32([
        [w * top_margin,         roi_start_y],
        [w * (1 - top_margin),   roi_start_y],
        [w,                      h],
        [0,                      h]
    ])


def bird_eye_view_zoom(frame_bgr, src_points, zoom_factor):
    h, w = frame_bgr.shape[:2]
    center_x = w // 2
    new_width = w * zoom_factor
    half_width = int(new_width // 2)
    dst_points = np.float32([
        [center_x - half_width, 0],
        [center_x + half_width, 0],
        [center_x + half_width, h],
        [center_x - half_width, h]
    ])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(frame_bgr, M, (w, h))


def remove_top_wall_noise(mask_binary):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        is_at_top = y < WALL_Y_LIMIT
        is_big_enough = (w > WALL_MIN_WIDTH or area > WALL_MIN_AREA)
        is_horizontal = w > (h * 1.5)
        if is_at_top and is_big_enough and is_horizontal:
            mask_binary[labels == i] = 0
    return mask_binary


def get_lane_mask(bev_bgr):
    blurred = cv2.GaussianBlur(bev_bgr, (5, 5), 0)
    mask_yellow = cv2.inRange(blurred, YELLOW_BGR_LOWER, YELLOW_BGR_UPPER)
    mask_white  = cv2.inRange(blurred, WHITE_BGR_LOWER,  WHITE_BGR_UPPER)

    kernel = np.ones((3, 3), np.uint8)
    mask_white  = remove_top_wall_noise(mask_white)
    mask_yellow = cv2.dilate(mask_yellow, kernel, iterations=1)

    combined_binary = cv2.bitwise_or(mask_yellow, mask_white)

    filled_view = np.zeros_like(bev_bgr)
    filled_view[mask_white == 255]  = (255, 255, 255)
    filled_view[mask_yellow == 255] = (0, 255, 255)
    return filled_view, combined_binary, mask_yellow


def find_all_peaks(histogram, threshold=50, min_dist=MIN_PEAK_DISTANCE):
    peaks = []
    candidates = np.where(histogram > threshold)[0]
    if len(candidates) == 0:
        return []
    current = [candidates[0]]
    for i in range(1, len(candidates)):
        if candidates[i] - candidates[i - 1] < min_dist:
            current.append(candidates[i])
        else:
            peaks.append(int(np.mean(current)))
            current = [candidates[i]]
    peaks.append(int(np.mean(current)))
    return peaks


def sliding_window_polyfit_multi(binary_warped):
    h, w = binary_warped.shape[:2]
    n_layers = 3
    layer_height = h // n_layers

    all_start_x = []
    for i in range(n_layers):
        y_low  = h - (i + 1) * layer_height
        y_high = h - i * layer_height
        histogram = np.sum(binary_warped[y_low:y_high, :], axis=0)
        peaks = find_all_peaks(histogram, threshold=50)
        for p in peaks:
            if all(abs(p - ex["x"]) >= MIN_PEAK_DISTANCE for ex in all_start_x):
                all_start_x.append({"x": p, "start_layer": i})

    detected_lines = []
    nonzero  = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    window_height = int(h // N_WINDOWS)

    for start_info in all_start_x:
        current_x   = start_info["x"]
        start_layer = start_info["start_layer"]
        lane_inds = []

        for window in range(start_layer, N_WINDOWS):
            win_y_low  = h - (window + 1) * window_height
            win_y_high = h - window * window_height
            win_x_low  = current_x - WINDOW_MARGIN
            win_x_high = current_x + WINDOW_MARGIN

            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            lane_inds.append(good_inds)

            if len(good_inds) > MIN_PIX:
                current_x = int(np.mean(nonzerox[good_inds]))

        lane_inds = np.concatenate(lane_inds) if len(lane_inds) else np.array([], dtype=int)
        if len(lane_inds) > 50:
            x = nonzerox[lane_inds]
            y = nonzeroy[lane_inds]
            try:
                fit = np.polyfit(y, x, 2)
                if abs(fit[0]) > MAX_CURVATURE:
                    linear_fit = np.polyfit(y, x, 1)
                    fit = np.array([0, linear_fit[0], linear_fit[1]])

                ploty = np.linspace(0, h - 1, h)
                fitx  = fit[0] * ploty**2 + fit[1] * ploty + fit[2]
                pts   = np.array([np.transpose(np.vstack([fitx, ploty]))])

                detected_lines.append({
                    "pts": pts.astype(int),
                    "base_x": start_info["x"],
                })
            except Exception:
                pass
    return detected_lines


def classify_line_type(binary_img, line_pts):
    h, w = binary_img.shape[:2]
    points = line_pts.reshape(-1, 2)
    total_points, white_points = 0, 0

    for i in range(0, len(points), 2):
        x, y = points[i]
        if 0 <= x < w and 0 <= y < h:
            total_points += 1
            roi = binary_img[max(0, y - 3):min(h, y + 4),
                             max(0, x - 3):min(w, x + 4)]
            if np.sum(roi) > 0:
                white_points += 1

    if total_points == 0:
        return "dashed"
    return "solid" if (white_points / total_points) > 0.6 else "dashed"


def determine_lane_status(center_x, left_line, right_line):
    if left_line is None and right_line is None:
        return "Undefined"

    margin = LANE_WIDTH_PX * 0.3
    dist_l = (center_x - left_line["base_x"]) if left_line else float("inf")
    dist_r = (right_line["base_x"] - center_x) if right_line else float("inf")
    type_l = left_line["type"] if left_line else None
    type_r = right_line["type"] if right_line else None

    status = "Undefined"
    if type_l == "yellow":
        status = "Lane 1"
        if dist_r < margin:
            status = "Between 1-2"
    elif type_l == "dashed" and type_r == "dashed":
        status = "Lane 2"
        if dist_l < margin:
            status = "Between 1-2"
        elif dist_r < margin:
            status = "Between 2-3"
    elif type_l == "dashed" and type_r == "solid":
        status = "Lane 3"
        if dist_l < margin:
            status = "Between 2-3"
    else:
        if type_l == "dashed" and type_r is None:
            status = "Between 1-2" if dist_l < margin else "Lane 2 (Est)"
        elif type_r == "dashed" and type_l is None:
            status = "Between 1-2" if dist_r < margin else "Lane 1 (Est)"
        elif type_r == "solid" and type_l is None:
            status = "Lane 3 (Est)"
    return status


def process_lane(frame_bgr):
    global last_lane_status, miss_count

    h, w = frame_bgr.shape[:2]
    my_center = w // 2

    src = get_roi_points(w, h, ROI_HEIGHT_RATIO_DETECT, TRAPEZOID_TOP_MARGIN_DETECT)
    bev = bird_eye_view_zoom(frame_bgr, src, BEV_ZOOM_FACTOR_DETECT)

    if BOTTOM_MASK_HEIGHT > 0:
        bev[-BOTTOM_MASK_HEIGHT:, :] = 0

    masked, binary, mask_yellow = get_lane_mask(bev)
    lines = sliding_window_polyfit_multi(binary)

    analyzed = []
    for line in lines:
        pts = line["pts"].reshape(-1, 2)
        yellow_hit = 0
        for i in range(0, len(pts), 10):
            px, py = pts[i]
            if 0 <= px < w and 0 <= py < h:
                roi = mask_yellow[max(0, py - 2):min(h, py + 3),
                                  max(0, px - 2):min(w, px + 3)]
                if np.sum(roi) > 0:
                    yellow_hit += 1

        if yellow_hit >= 3:
            l_type  = "yellow"
            l_color = (0, 255, 0)
        else:
            l_type  = classify_line_type(binary, line["pts"])
            l_color = (255, 0, 0) if l_type == "dashed" else (255, 0, 255)

        line["type"]  = l_type
        line["color"] = l_color
        analyzed.append(line)

    left_candidates  = [ln for ln in analyzed if ln["base_x"] < my_center]
    right_candidates = [ln for ln in analyzed if ln["base_x"] >= my_center]
    left_candidates.sort(key=lambda x: x["base_x"], reverse=True)
    right_candidates.sort(key=lambda x: x["base_x"])

    left  = left_candidates[0]  if left_candidates  else None
    right = right_candidates[0] if right_candidates else None

    current_status = determine_lane_status(my_center, left, right)

    if current_status != "Undefined":
        last_lane_status = current_status
        miss_count = 0
    else:
        miss_count += 1
        current_status = (last_lane_status + " (Mem)") if miss_count < 10 else "Undefined"

    final_view = masked.copy()
    for ln in analyzed:
        cv2.polylines(final_view, ln["pts"], False, ln["color"], 3)

    cv2.putText(final_view, current_status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.line(final_view, (my_center, 0), (my_center, h), (0, 0, 255), 1)

    tx = {"status_id": status_to_id(current_status), "status_str": current_status}
    return final_view, tx


# ==========================================
# Thread 1: Capture (max fps)
# ==========================================
def capture_thread(picam2):
    while not _stop_event.is_set():
        frame = picam2.capture_array()  # libcamera 설정에 따라 RGB/BGR
        set_latest_frame(frame)
        if CAPTURE_SLEEP > 0:
            time.sleep(CAPTURE_SLEEP)


# ==========================================
# Thread 2: YOLO (Hailo HEF 사용)
# ==========================================
def yolo_thread():
    print("[HYOLO] thread start")

    # 1) HEF 로드
    try:
        print(f"[HYOLO] Loading HEF: {HAILO_HEF_PATH}")
        hef = HEF(HAILO_HEF_PATH)
    except Exception as e:
        print(f"[HYOLO] HEF load FAILED: {repr(e)}")
        return

    # 2) Hailo 디바이스 / VDevice / NetworkGroup 설정
    devices = Device.scan()
    if not devices:
        print("[HYOLO] No Hailo devices found.")
        return

    try:
        vdev = VDevice(device_ids=devices)
        cfg_params = ConfigureParams.create_from_hef(
            hef,
            interface=HailoStreamInterface.PCIe
        )
        network_group = vdev.configure(hef, cfg_params)[0]
        ng_params = network_group.create_params()
    except Exception as e:
        print(f"[HYOLO] configure FAILED: {repr(e)}")
        return

    in_info  = hef.get_input_vstream_infos()[0]
    out_info = hef.get_output_vstream_infos()[0]

    print(f"[HYOLO] Input vstream : name={in_info.name}, shape={in_info.shape}")
    print(f"[HYOLO] Output vstream: name={out_info.name}, shape={out_info.shape}")

    in_shape = in_info.shape
    if len(in_shape) == 3:
        in_h, in_w = in_shape[0], in_shape[1]
    elif len(in_shape) == 4:
        in_h, in_w = in_shape[1], in_shape[2]
    else:
        print(f"[HYOLO] Unexpected input shape: {in_shape}")
        return

    in_params = InputVStreamParams.make_from_network_group(
        network_group,
        quantized=False,
        format_type=FormatType.FLOAT32,
    )
    out_params = OutputVStreamParams.make_from_network_group(
        network_group,
        quantized=False,
        format_type=FormatType.FLOAT32,
    )

    t0 = time.time()
    frames = 0
    debug_printed = False

    print("[HYOLO] Ready. Running inference loop...")

    with InferVStreams(
        network_group,
        in_params,
        out_params,
        tf_nms_format=True,   # (num_classes, [ymin,xmin,ymax,xmax,score], num_dets)
    ) as infer_pipeline:
        with network_group.activate(ng_params):
            while not _stop_event.is_set():
                frame = get_latest_frame_copy()
                if frame is None:
                    time.sleep(0.01)
                    continue

                h, w = frame.shape[:2]

                # 입력 크기에 맞춰 리사이즈
                resized = cv2.resize(frame, (in_w, in_h))

                input_tensor = resized.astype(np.float32) / 255.0
                input_data = {
                    in_info.name: np.expand_dims(input_tensor, axis=0)
                }

                try:
                    out_data = infer_pipeline.infer(input_data)
                except Exception as e:
                    print(f"[HYOLO] infer FAILED: {repr(e)}")
                    time.sleep(0.01)
                    continue

                frames += 1
                dt = time.time() - t0
                fps = (frames / dt) if dt > 0 else 0.0

                boxes = []
                confidences = []
                class_ids = []

                for key, tensor in out_data.items():
                    arr = tensor[0]  # (num_classes, 5, num_dets) 가정
                    if arr.ndim != 3:
                        if not debug_printed:
                            print(f"[HYOLO] Unexpected out ndim={arr.ndim}, shape={arr.shape}")
                        continue

                    num_classes, bbox_params, num_dets = arr.shape
                    if bbox_params < 5:
                        if not debug_printed:
                            print(f"[HYOLO] bbox_params < 5, shape={arr.shape}")
                        continue

                    if not debug_printed:
                        conf_map = arr[:, 4, :]
                        print(f"[HYOLO] conf min={conf_map.min():.6f}, max={conf_map.max():.6f}")
                        sample = arr[0, :, 0]
                        print(f"[HYOLO] sample bbox (class 0, det 0) = {sample}")
                        debug_printed = True

                    # [ymin, xmin, ymax, xmax, score]
                    for cls_id in range(min(num_classes, len(CLASS_NAMES))):
                        for det_id in range(num_dets):
                            bbox = arr[cls_id, :, det_id]
                            conf = float(bbox[4])
                            if conf < HAILO_CONF_THRESHOLD:
                                continue

                            y1n, x1n, y2n, x2n = bbox[0], bbox[1], bbox[2], bbox[3]
                            x1 = int(x1n * w)
                            y1 = int(y1n * h)
                            x2 = int(x2n * w)
                            y2 = int(y2n * h)

                            if x2 <= x1 or y2 <= y1:
                                continue

                            boxes.append([x1, y1, x2, y2])
                            confidences.append(conf)
                            class_ids.append(cls_id)

                top_objs = []

                # YOLO 쪽은 현재 색이 정상이라고 했으니 frame 그대로 사용
                frame_draw = frame.copy()

                if boxes:
                    boxes_np = np.array(boxes)
                    confs_np = np.array(confidences)
                    clss_np  = np.array(class_ids)

                    order = np.argsort(-confs_np)
                    if len(order) > HAILO_MAX_BOXES:
                        order = order[:HAILO_MAX_BOXES]

                    boxes_np = boxes_np[order]
                    confs_np = confs_np[order]
                    clss_np  = clss_np[order]

                    for (x1, y1, x2, y2), conf, cid in zip(boxes_np, confs_np, clss_np):
                        cid  = int(cid)
                        conf = float(conf)
                        name = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else f"id{cid}"

                        color = (0, 255, 0)
                        cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame_draw, f"{name}:{conf:.2f}",
                                    (x1, max(y1 - 10, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        xc = float((x1 + x2) * 0.5)
                        yc = float((y1 + y2) * 0.5)

                        top_objs.append({
                            "id": cid,
                            "name": name,
                            "conf": conf,
                            "xc": xc,
                            "yc": yc
                        })

                cv2.putText(frame_draw, f"HYOLO FPS: {fps:.2f} objs:{len(top_objs)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)

                set_yolo_frame(frame_draw)
                set_yolo_tx(top_objs)


# ==========================================
# Thread 3: Lane processing + store tx (fixed rate)
# ==========================================
def lane_thread():
    period = 1.0 / max(LANE_PROCESS_HZ, 0.1)
    next_t = time.time()

    while not _stop_event.is_set():
        now = time.time()
        if now < next_t:
            time.sleep(min(0.005, next_t - now))
            continue
        next_t += period

        frame = get_latest_frame_copy()
        if frame is None:
            continue

        # YOLO 쪽 색이 기준과 같다고 가정하고, frame 을 BGR 취급
        bgr = frame.copy()
        bgr = apply_gamma(bgr, gamma=GAMMA_VALUE)

        # BEV / 마스크 / 슬라이딩 윈도우
        final_view, tx = process_lane(bgr)
        set_lane_final(final_view)
        set_lane_tx(tx)

# ==========================================
# Thread 4: UART sender (Binary Packet)
# ==========================================
def uart_thread(ser):
    if ser is None:
        return

    lane_period = 1.0 / max(LANE_SEND_HZ, 0.1)
    obj_period  = 1.0 / max(OBJ_SEND_HZ,  0.1)

    next_lane_t = time.time()
    next_obj_t  = time.time()

    while not _stop_event.is_set():
        now = time.time()
        sleep_t = min(next_lane_t, next_obj_t) - now
        if sleep_t > 0:
            time.sleep(min(0.005, sleep_t))
            continue

        ts_ms = int(time.time() * 1000) & 0xFFFFFFFF

        # ---- lane packet @ 1Hz ----
        if now >= next_lane_t:
            next_lane_t += lane_period

            lane = get_lane_tx()
            lane_id = 0
            if lane is not None:
                raw_id = lane.get("status_id", 0)
                # 간단 매핑: 1,2 -> 1 / 3 ->2 / 4,5 ->3
                if   raw_id == 1: lane_id = 1
                elif raw_id == 2: lane_id = 1
                elif raw_id == 3: lane_id = 2
                elif raw_id == 4: lane_id = 3
                elif raw_id == 5: lane_id = 3
                else:             lane_id = 0

            try:
                packet_lane = struct.pack('<BBI', 0xA1, lane_id, ts_ms)
                ser.write(packet_lane)
            except Exception as e:
                print(f"[UART] Lane send err: {e}")

        # ---- object packets @ 20Hz ----
        if now >= next_obj_t:
            next_obj_t += obj_period

            objs = get_yolo_tx() or []
            for o in objs[:MAX_SEND_OBJECTS]:
                cls_name = str(o.get("name", "")).lower()
                cls_id   = int(o.get("id", -1))

                # 기본적으로 1을 보냄
                obj_type = 1
                # 라바콘 인식하면 2보냄
                if "rubber_cone" in cls_name or "cone" in cls_name or cls_id == 5:
                    obj_type = 2

                yc = float(o.get("yc", 0.0))
                # 화면 하단에 가까울수록 dist 작게, 단순 비례
                dist_m = int(max(1, (IMG_HEIGHT - yc) / 20))  # 832 기준 대략 비례값
                dist_m = min(dist_m, 255)

                try:
                    packet_obj = struct.pack('<BBBI', 0xA2, obj_type, dist_m, ts_ms)
                    ser.write(packet_obj)
                except Exception as e:
                    print(f"[UART] Obj send err: {e}")


# ==========================================
# Main
# ==========================================
def main():
    global _seq
    _seq = 0

    # Picamera2 import
    try:
        from picamera2 import Picamera2
        from libcamera import Transform
    except ImportError:
        print("Error: Picamera2 or libcamera not found.")
        return

    # UART open
    try:
        import serial
        ser = serial.Serial(UART_DEVICE, UART_BAUD, timeout=0)
        print(f"UART open OK: {UART_DEVICE} @ {UART_BAUD}")
    except Exception as e:
        print(f"UART open FAILED: {e}")
        ser = None

    # Camera: 832x832 RGB888, 180도 회전
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (IMG_WIDTH, IMG_HEIGHT), "format": "RGB888"},
        transform=Transform(rotation=180)
    )
    picam2.configure(config)
    picam2.start()

    th_cap  = threading.Thread(target=capture_thread, args=(picam2,), daemon=True)
    th_yolo = threading.Thread(target=yolo_thread, daemon=True)
    th_lane = threading.Thread(target=lane_thread, daemon=True)
    th_uart = threading.Thread(target=uart_thread, args=(ser,), daemon=True)

    th_cap.start()
    th_yolo.start()
    th_lane.start()
    th_uart.start()

    print("Running: Capture(832x832) + Hailo YOLO + Lane + UART(binary). Press 'q' to quit.")

    try:
        while True:
            # 최종 차선 결과
            lane_final = get_lane_final()
            if lane_final is not None:
                cv2.imshow("Lane Final", lane_final)

            # YOLO 결과
            yolo_img = get_yolo_frame()
            if yolo_img is not None:
                cv2.imshow("YOLO", yolo_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.002)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        _stop_event.set()

        try:
            th_cap.join(timeout=1.0)
            th_yolo.join(timeout=1.0)
            th_lane.join(timeout=1.0)
            th_uart.join(timeout=1.0)
        except Exception:
            pass

        try:
            picam2.stop()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            if ser is not None:
                ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
