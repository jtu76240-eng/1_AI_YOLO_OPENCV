import cv2
import numpy as np
import time

# ==========================================
# [설정값]
# ==========================================
SWAP_RGB_TO_BGR = False

# 카메라 / 처리 해상도 832x832
IMG_WIDTH, IMG_HEIGHT = 832, 832

# ---------------------------
# ROI / BEV / GAMMA 
# ---------------------------
#[TUNE] ROI=0.57, TOP=0.44, ZOOM=0.27, GAMMA=1.70
ROI_HEIGHT_RATIO   = 0.60
TRAPEZOID_TOP_MARGIN = 0.44
BEV_ZOOM_FACTOR    = 0.27
GAMMA_VALUE        = 1.70

BOTTOM_MASK_HEIGHT = 0

# ---------------------------
# 색상 임계값
# ---------------------------
YELLOW_BGR_LOWER = np.array([70, 150, 210])
YELLOW_BGR_UPPER = np.array([150, 230, 255])
WHITE_BGR_LOWER  = np.array([130, 130, 130])
WHITE_BGR_UPPER  = np.array([255, 255, 255])

# ---------------------------
# 해상도 의존 파라미터 
# ---------------------------
#[TUNE] WIN_MARGIN=150, MIN_PIX=118, N_WIN=15, LANE_W=210
N_WINDOWS        = 15
WINDOW_MARGIN    = 150      
MIN_PIX          = 118     
LANE_WIDTH_PX    = 210     

WALL_Y_LIMIT     = 173      
WALL_MIN_WIDTH   = 62     
WALL_MIN_AREA    = 1800  

MIN_PEAK_DISTANCE = 156     
MAX_CURVATURE     = 0.010

last_lane_status = "Undefined"
miss_count = 0

# ---------------------------
# Gamma LUT (속도 최적화)
# ---------------------------
_gamma_cache = {"val": None, "lut": None}
def apply_gamma(image, gamma=1.0):
    global _gamma_cache
    if _gamma_cache["val"] != gamma:
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        _gamma_cache["val"] = gamma
        _gamma_cache["lut"] = table
    return cv2.LUT(image, _gamma_cache["lut"])

def get_roi_points(w, h):
    roi_h = int(h * ROI_HEIGHT_RATIO)
    roi_start_y = h - roi_h
    src_points = np.float32([
        [w * TRAPEZOID_TOP_MARGIN,          roi_start_y],
        [w * (1 - TRAPEZOID_TOP_MARGIN),    roi_start_y],
        [w,                                 h],
        [0,                                 h]
    ])
    return src_points

def bird_eye_view_zoom(frame, src_points):
    h, w = frame.shape[:2]
    center_x = w // 2
    new_width = w * BEV_ZOOM_FACTOR
    half_width = int(new_width // 2)

    dst_points = np.float32([
        [center_x - half_width, 0],
        [center_x + half_width, 0],
        [center_x + half_width, h],
        [center_x - half_width, h]
    ])

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(frame, M, (w, h))
    return warped

def remove_top_wall_noise(mask_binary):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        is_at_top     = y < WALL_Y_LIMIT
        is_big_enough = (w > WALL_MIN_WIDTH or area > WALL_MIN_AREA)
        is_horizontal = w > (h * 1.5)
        if is_at_top and is_big_enough and is_horizontal:
            mask_binary[labels == i] = 0
    return mask_binary

def get_lane_mask(bev_frame):
    blurred = cv2.GaussianBlur(bev_frame, (5, 5), 0)

    mask_yellow = cv2.inRange(blurred, YELLOW_BGR_LOWER, YELLOW_BGR_UPPER)
    mask_white  = cv2.inRange(blurred, WHITE_BGR_LOWER, WHITE_BGR_UPPER)

    kernel = np.ones((3,3), np.uint8)

    mask_white  = remove_top_wall_noise(mask_white)
    mask_yellow = cv2.dilate(mask_yellow, kernel, iterations=1)

    combined_binary = cv2.bitwise_or(mask_yellow, mask_white)

    filled_view = np.zeros_like(bev_frame)
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
        if candidates[i] - candidates[i-1] < min_dist:
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
        peaks = find_all_peaks(histogram)
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
        lane_inds   = []

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

                ploty = np.linspace(0, h-1, h)
                fitx  = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
                pts   = np.array([np.transpose(np.vstack([fitx, ploty]))])

                detected_lines.append({"pts": pts.astype(int), "base_x": start_info["x"]})
            except:
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
            roi = binary_img[max(0, y-3):min(h, y+4), max(0, x-3):min(w, x+4)]
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

def process_lane(frame):
    global last_lane_status, miss_count
    h, w = frame.shape[:2]
    my_center = w // 2

    src_points = get_roi_points(w, h)
    bev_view   = bird_eye_view_zoom(frame, src_points)

    if BOTTOM_MASK_HEIGHT > 0:
        bev_view[-BOTTOM_MASK_HEIGHT:, :] = 0

    masked_view, combined_binary, mask_yellow = get_lane_mask(bev_view)
    final_view = masked_view.copy()

    all_lines = sliding_window_polyfit_multi(combined_binary)
    analyzed_lines = []

    for line in all_lines:
        is_yellow = False
        line_points = line["pts"].reshape(-1, 2)

        yellow_hit_count = 0
        check_step = 10
        for i in range(0, len(line_points), check_step):
            px, py = line_points[i]
            if 0 <= px < w and 0 <= py < h:
                roi = mask_yellow[max(0, py-2):min(h, py+3), max(0, px-2):min(w, px+3)]
                if np.sum(roi) > 0:
                    yellow_hit_count += 1
        if yellow_hit_count >= 3:
            is_yellow = True

        if is_yellow:
            l_type  = "yellow"
            l_color = (0, 255, 0)
        else:
            l_type  = classify_line_type(combined_binary, line["pts"])
            l_color = (255, 0, 0) if l_type == "dashed" else (255, 0, 255)

        line["type"]  = l_type
        line["color"] = l_color
        analyzed_lines.append(line)

    left_candidates  = [ln for ln in analyzed_lines if ln["base_x"] < my_center]
    right_candidates = [ln for ln in analyzed_lines if ln["base_x"] >= my_center]
    left_candidates.sort(key=lambda x: x["base_x"], reverse=True)
    right_candidates.sort(key=lambda x: x["base_x"])

    closest_left  = left_candidates[0]  if left_candidates  else None
    closest_right = right_candidates[0] if right_candidates else None

    current_status = determine_lane_status(my_center, closest_left, closest_right)

    if current_status != "Undefined":
        last_lane_status = current_status
        miss_count = 0
    else:
        miss_count += 1
        current_status = (last_lane_status + " (Mem)") if miss_count < 10 else "Undefined"

    for line in analyzed_lines:
        cv2.polylines(final_view, line["pts"], False, line["color"], 3)

    cv2.putText(final_view, current_status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.line(final_view, (my_center, 0), (my_center, h), (0, 0, 255), 1)

    pts = np.array(src_points, np.int32).reshape((-1, 1, 2))
    display_frame = frame.copy()
    cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)

    return display_frame, bev_view, masked_view, final_view

# =========================================================
# Trackbar UI
# =========================================================
def _noop(x): pass

def setup_trackbars():
    cv2.namedWindow("Tuning", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tuning", 420, 300)

    cv2.createTrackbar("ROI% (50~90)",   "Tuning", int(ROI_HEIGHT_RATIO * 100),   90, _noop)
    cv2.createTrackbar("TopMargin% (10~49)", "Tuning", int(TRAPEZOID_TOP_MARGIN * 100), 49, _noop)
    cv2.createTrackbar("BEVZoom% (10~90)",   "Tuning", int(BEV_ZOOM_FACTOR * 100),     90, _noop)
    cv2.createTrackbar("Gamma x10 (5~40)",   "Tuning", int(GAMMA_VALUE * 10),          40, _noop)

    # 832 해상도에 맞춰 상한 늘려줌
    cv2.createTrackbar("WinMargin", "Tuning", WINDOW_MARGIN,   400, _noop)
    cv2.createTrackbar("MinPix",    "Tuning", MIN_PIX,         300, _noop)
    cv2.createTrackbar("NWindows",  "Tuning", N_WINDOWS,        30, _noop)
    cv2.createTrackbar("LaneWidth", "Tuning", LANE_WIDTH_PX,   400, _noop)

def read_trackbars():
    global ROI_HEIGHT_RATIO, TRAPEZOID_TOP_MARGIN, BEV_ZOOM_FACTOR, GAMMA_VALUE
    global WINDOW_MARGIN, MIN_PIX, N_WINDOWS, LANE_WIDTH_PX

    roi  = cv2.getTrackbarPos("ROI% (50~90)",     "Tuning")
    top  = cv2.getTrackbarPos("TopMargin% (10~49)", "Tuning")
    zoom = cv2.getTrackbarPos("BEVZoom% (10~90)", "Tuning")
    gam  = cv2.getTrackbarPos("Gamma x10 (5~40)", "Tuning")

    ROI_HEIGHT_RATIO   = max(0.50, roi / 100.0)
    TRAPEZOID_TOP_MARGIN = min(0.49, max(0.10, top / 100.0))
    BEV_ZOOM_FACTOR    = min(0.90, max(0.10, zoom / 100.0))
    GAMMA_VALUE        = min(4.0,  max(0.5, gam / 10.0))

    WINDOW_MARGIN  = cv2.getTrackbarPos("WinMargin", "Tuning")
    MIN_PIX        = cv2.getTrackbarPos("MinPix",    "Tuning")
    N_WINDOWS      = max(5, cv2.getTrackbarPos("NWindows", "Tuning"))
    LANE_WIDTH_PX  = cv2.getTrackbarPos("LaneWidth", "Tuning")

def main():
    try:
        from picamera2 import Picamera2
        from libcamera import Transform
    except ImportError:
        print("Error: Picamera2/libcamera not found.")
        return

    setup_trackbars()

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (IMG_WIDTH, IMG_HEIGHT), "format": "RGB888"},
        transform=Transform(rotation=180)  # 필요 없으면 삭제
    )
    picam2.configure(config)
    picam2.start()

    print("Lane Baseline + Live Tuning (832x832). Press 'q' to exit.")
    last_print = time.time()

    try:
        while True:
            read_trackbars()

            frame = picam2.capture_array()
            if SWAP_RGB_TO_BGR:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame = apply_gamma(frame, gamma=GAMMA_VALUE)

            frame_ori, frame_bev, frame_clean, frame_final = process_lane(frame)

            cv2.imshow("1. Original",          frame_ori)
            cv2.imshow("2. BEV",               frame_bev)
            cv2.imshow("3. Lane Mask (Clean)", frame_clean)
            cv2.imshow("4. Lane Mask (Final)", frame_final)

            # 1초마다 현재 튜닝값 출력
            if time.time() - last_print > 1.0:
                last_print = time.time()
                print(f"[TUNE] ROI={ROI_HEIGHT_RATIO:.2f}, TOP={TRAPEZOID_TOP_MARGIN:.2f}, "
                      f"ZOOM={BEV_ZOOM_FACTOR:.2f}, GAMMA={GAMMA_VALUE:.2f}, "
                      f"WIN_MARGIN={WINDOW_MARGIN}, MIN_PIX={MIN_PIX}, N_WIN={N_WINDOWS}, "
                      f"LANE_W={LANE_WIDTH_PX}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
