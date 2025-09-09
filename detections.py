import cv2
import numpy as np
from helpers import is_intersecting
import config

VALID_COLOR = (0, 255, 0)
INVALID_COLOR = (0, 0, 255)

def find_rectangles(lines, image):
    """Find rectangles in a list of contours based on area threshold."""
    if lines is None or len(lines) == 0:
        img_height, img_width = image.shape[:2]
        w = int(img_width/2)
        cv2.putText(image, "No Bounding Box found", (w, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 255), 2, cv2.LINE_AA)
        config.NUMBER_OF_ERRORS += 1
        return [],[]

    img_height, img_width = image.shape[:2]
    image_area = img_width * img_height
    w = int(img_width/2)
    horizontal_lines = [line for line in lines if abs(line[0][1] - line[0][3]) < 1]

    if len(horizontal_lines) < 2:
        cv2.putText(image, "No Bounding Box found", (w, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 255), 2, cv2.LINE_AA)
        config.NUMBER_OF_ERRORS += 1
        return [],[]

    horizontal_lines = sorted(horizontal_lines, key=lambda line: line[0][1])
    grouped_lines = []
    current_group = []

    for line in horizontal_lines:
        x1, y1, x2, y2 = line[0]
        if not current_group:
            current_group.append(line)
            continue
        _, last_y1, _, last_y2 = current_group[-1][0]
        if abs(y1 - last_y1) <= 10 or abs(y2 - last_y2) <= 10:
            current_group.append(line)
        else:
            grouped_lines.append(current_group)
            current_group = [line]
    if current_group:
        grouped_lines.append(current_group)

    merged_lines = []
    for group in grouped_lines:
        x1_min = min(line[0][0] for line in group)
        y1_min = min(line[0][1] for line in group)
        x2_max = max(line[0][2] for line in group)
        y2_max = max(line[0][3] for line in group)
        merged_lines.append((x1_min, y1_min, x2_max, y2_max))

    if len(merged_lines) < 2:
        cv2.putText(image, "No Bounding Box found", (w, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 255), 2, cv2.LINE_AA)
        config.NUMBER_OF_ERRORS += 1
        return [],[]

    longest_horizontal = sorted(merged_lines, key=lambda line: line[2] - line[0], reverse=True)[:2]
    top_left = (longest_horizontal[0][0], longest_horizontal[0][1])
    bottom_left = (longest_horizontal[0][0], longest_horizontal[1][1])
    bottom_right = (longest_horizontal[1][2], longest_horizontal[1][3])
    top_right = (longest_horizontal[1][2], longest_horizontal[0][3])

    if top_left[1] > bottom_left[1]:
        top_left, bottom_left = bottom_left, top_left
        top_right, bottom_right = bottom_right, top_right

    rect_width = bottom_right[0] - bottom_left[0]
    rect_height = bottom_left[1] - top_left[1]
    rect_area = rect_width * rect_height

    if rect_area < 0.3 * image_area:
        cv2.putText(image, "No Bounding Box found", (w, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return [],[]

    points = np.array([
        [top_left], [bottom_left], [bottom_right], [top_right]
    ], dtype=np.int32)
    rectangles = [top_left, bottom_left, bottom_right, top_right]
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=3)
    return [points], rectangles

def process_stickmen_detections(results, rectangles, image):
    """Process detected stickmen and filter out those intersecting with rectangles."""
    stickman_count = 0
    stickmen_bboxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if any(is_intersecting(rect, (x1, y1, x2, y2)) for rect in rectangles):
                continue
            cv2.rectangle(image, (x1, y1), (x2, y2), VALID_COLOR, 2)
            stickman_count += 1
            stickmen_bboxes.append((x1, y1, x2, y2))
    return stickman_count, stickmen_bboxes

def detect_arrows(model, image):
    """Detect arrows in an image and filter out those inside text bounding boxes and overlapping arrows."""
    results = model(image)
    result = results[0]
    detected_arrows = []
    if result.masks is None or result.boxes is None:
        print("No arrows detected.")
        return []
    potential_arrows = []
    for mask, box in zip(result.masks.data, result.boxes.xyxy):
        mask = mask.cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        potential_arrows.append((x1, y1, x2, y2))
    for i, arrow in enumerate(potential_arrows):
        should_keep = True
        x1, y1, x2, y2 = arrow
        for kept_arrow in detected_arrows:
            kx1, ky1, kx2, ky2 = kept_arrow
            if is_overlapping((x1, y1, x2, y2), (kx1, ky1, kx2, ky2)):
                should_keep = False
                break
        if should_keep:
            config.ARROW_LINES += 1
            detected_arrows.append((x1, y1, x2, y2))
            cv2.rectangle(image, (x1, y1), (x2, y2), VALID_COLOR, 2)
    return detected_arrows

def is_overlapping(box1, box2):
    """Check if two bounding boxes overlap."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    if x2_1 < x1_2 or x2_2 < x1_1:
        return False
    if y2_1 < y1_2 or y2_2 < y1_1:
        return False
    return True

def detect_arrow_head(image, arrow_head_model):
    """Detect arrow heads in the image."""
    results = arrow_head_model(image)
    detected_arrow_heads = []
    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(image, (x1, y1), (x2, y2), VALID_COLOR, 2)
            detected_arrow_heads.append((x1, y1, x2, y2))
            config.ARROW_HEADS += 1
    return detected_arrow_heads

def detect_ovals(model, image, rectangles):
    """Detect ovals and check if they are inside rectangles."""
    results = model(image)
    ovals_detected = []
    isValid = True
    for result in results:
        for box in result.boxes:
            if box.conf[0] > 0.75:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ovals_detected.append((x1, y1, x2, y2))
                is_inside = any(is_intersecting(rect, (x1, y1, x2, y2)) for rect in rectangles)
                color = VALID_COLOR if is_inside else INVALID_COLOR
                if color == VALID_COLOR:
                    config.VALID_OVALS += 1
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                else:
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, "Oval is not within boundary box", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    isValid = False
                    config.NUMBER_OF_ERRORS += 1
    return ovals_detected

def detect_line_to_ovals(image, line_to_oval):
    """Detect lines connecting ovals and mark them as broken links."""
    results = line_to_oval(image)
    detected_lines = []
    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(image, (x1, y1), (x2, y2), INVALID_COLOR, 2)
            cv2.putText(image, "Links between use cases must be broken", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INVALID_COLOR, 2)
            config.NUMBER_OF_ERRORS += 1
            detected_lines.append((x1, y1, x2, y2))
    return detected_lines

def detect_line(image, line_model):
    """Detect lines in the image using a segmentation model."""
    results = line_model(image)
    detected_masks = []
    img_height, img_width = image.shape[:2]
    for result in results:
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks
            for i, mask_tensor in enumerate(masks.data):
                mask = mask_tensor.cpu().numpy()
                mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
                binary_mask = (mask > 0.5).astype(np.uint8) * 255
                detected_masks.append(binary_mask)
    return detected_masks, image

def get_line_endpoints(line_masks):
    """Extract the leftmost and rightmost points from line masks."""
    if isinstance(line_masks, list):
        endpoints = []
        for mask in line_masks:
            non_zero_points = np.where(mask > 0)
            if len(non_zero_points[0]) == 0:
                endpoints.append((None, None))
                continue
            points = np.column_stack((non_zero_points[0], non_zero_points[1]))
            sorted_points = points[points[:, 1].argsort()]
            start_point = (int(sorted_points[0, 1]), int(sorted_points[0, 0]))
            end_point = (int(sorted_points[-1, 1]), int(sorted_points[-1, 0]))
            endpoints.append((start_point, end_point))
        return endpoints
    else:
        non_zero_points = np.where(line_masks > 0)
        if len(non_zero_points[0]) == 0:
            return None, None
        points = np.column_stack((non_zero_points[0], non_zero_points[1]))
        sorted_points = points[points[:, 1].argsort()]
        start_point = (int(sorted_points[0, 1]), int(sorted_points[0, 0]))
        end_point = (int(sorted_points[-1, 1]), int(sorted_points[-1, 0]))
        return start_point, end_point

def does_line_intersect(image, ovalbbox, stickmanbbox, line_masks):
    """Check if line masks properly connect oval and stickman bounding boxes."""
    def enlarge_bbox(bbox, width_factor=2, height_factor=1.0):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        width_expand = (width * width_factor - width) / 2
        height_expand = (height * height_factor - height) / 2
        return (
            int(x1 - width_expand),
            int(y1 - height_expand),
            int(x2 + width_expand),
            int(y2 + height_expand)
        )
    def point_in_bbox(point, bbox):
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2
    def mask_bbox_intersect(mask, bbox):
        x1, y1, x2, y2 = bbox
        h, w = mask.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x1 >= x2 or y1 >= y2:
            return False, 0
        mask_region = mask[y1:y2, x1:x2]
        intersection_area = np.sum(mask_region > 0)
        intersects = intersection_area > 0
        return intersects, intersection_area
    def get_line_endpoints(line_mask):
        non_zero_points = np.where(line_mask > 0)
        if len(non_zero_points[0]) == 0:
            return None, None
        points = np.column_stack((non_zero_points[0], non_zero_points[1]))
        sorted_points = points[points[:, 1].argsort()]
        start_point = (int(sorted_points[0, 1]), int(sorted_points[0, 0]))
        end_point = (int(sorted_points[-1, 1]), int(sorted_points[-1, 0]))
        return start_point, end_point
    enlarged_stickmanbbox = [enlarge_bbox(bbox, width_factor=2.5) for bbox in stickmanbbox]
    enlarged_ovalbbox = [enlarge_bbox(bbox, width_factor=1.1, height_factor=1.1) for bbox in ovalbbox]
    stickman_connected = [False] * len(stickmanbbox)
    oval_connections = {}
    line_connections = {}
    for line_idx, line_mask in enumerate(line_masks):
        stickman_connections = []
        oval_connections_for_line = []
        start_point, end_point = get_line_endpoints(line_mask)
        if not start_point or not end_point:
            continue
        start_in_stickman = None
        end_in_stickman = None
        start_in_oval = None
        end_in_oval = None
        for stickman_idx, stickman in enumerate(enlarged_stickmanbbox):
            if point_in_bbox(start_point, stickman):
                start_in_stickman = stickman_idx
                stickman_connections.append((stickman_idx, 1))
            if point_in_bbox(end_point, stickman):
                end_in_stickman = stickman_idx
                if stickman_idx not in [s[0] for s in stickman_connections]:
                    stickman_connections.append((stickman_idx, 1))
        for oval_idx, oval in enumerate(enlarged_ovalbbox):
            if point_in_bbox(start_point, oval):
                start_in_oval = oval_idx
                oval_connections_for_line.append((oval_idx, 1))
                if oval_idx not in oval_connections:
                    oval_connections[oval_idx] = []
                oval_connections[oval_idx].append(line_idx)
            if point_in_bbox(end_point, oval):
                end_in_oval = oval_idx
                if oval_idx not in [o[0] for o in oval_connections_for_line]:
                    oval_connections_for_line.append((oval_idx, 1))
                    if oval_idx not in oval_connections:
                        oval_connections[oval_idx] = []
                    oval_connections[oval_idx].append(line_idx)
        connects_stickman_to_oval = False
        if start_in_stickman is not None and end_in_oval is not None:
            connects_stickman_to_oval = True
            stickman_connected[start_in_stickman] = True
        elif start_in_oval is not None and end_in_stickman is not None:
            connects_stickman_to_oval = True
            stickman_connected[end_in_stickman] = True
        if not connects_stickman_to_oval:
            for stickman_idx, stickman in enumerate(enlarged_stickmanbbox):
                intersects, area = mask_bbox_intersect(line_mask, stickman)
                if intersects and (stickman_idx not in [s[0] for s in stickman_connections]):
                    stickman_connections.append((stickman_idx, area))
            for oval_idx, oval in enumerate(enlarged_ovalbbox):
                intersects, area = mask_bbox_intersect(line_mask, oval)
                if intersects and (oval_idx not in [o[0] for o in oval_connections_for_line]):
                    oval_connections_for_line.append((oval_idx, area))
                    if oval_idx not in oval_connections:
                        oval_connections[oval_idx] = []
                    oval_connections[oval_idx].append(line_idx)
        connections = []
        for stickman_idx, area in stickman_connections:
            connections.append(("stickman", stickman_idx, area))
        for oval_idx, area in oval_connections_for_line:
            connections.append(("oval", oval_idx, area))
        line_connections[line_idx] = connections
        if oval_connections_for_line and stickman_connections:
            for stickman_idx, _ in stickman_connections:
                stickman_connected[stickman_idx] = True
    for stickman_idx, is_connected in enumerate(stickman_connected):
        if not is_connected:
            x1, y1, x2, y2 = stickmanbbox[stickman_idx]
            text_pos = (int((x1 + x2) / 2), int(y1 - 10))
            cv2.putText(image, "Stickman not connected to any ovals", text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            config.NUMBER_OF_ERRORS += 1
    for line_mask in line_masks:
        color_overlay = np.zeros_like(image, dtype=np.uint8)
        color_overlay[line_mask > 0] = [0, 255, 255]
        image = cv2.addWeighted(image, 1, color_overlay, 0.3, 0)

def is_intersecting_arrows(rect1, rect2):
    """Check if two rectangles intersect."""
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

def get_line_edges(image_path, min_line_length=100, max_vertical_deviation=15, max_horizontal_deviation=0):
    """Detect and filter lines in an image using Hough Transform."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    edges = cv2.Canny(thresh, 50, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    line_coordinates = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if abs(x2 - x1) > max_vertical_deviation and abs(y2 - y1) > max_horizontal_deviation and line_length >= min_line_length:
                line_coordinates.append(((x1, y1), (x2, y2)))
    return line_coordinates