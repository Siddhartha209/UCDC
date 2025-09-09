import cv2
import config
import numpy as np
from ultralytics import YOLO
from detections import (
    find_rectangles, process_stickmen_detections, detect_arrows, is_overlapping,
    detect_arrow_head, detect_ovals, detect_line_to_ovals, detect_line,
    get_line_endpoints, does_line_intersect, is_intersecting_arrows, get_line_edges
)
from text_utils import (
    is_valid_word, extract_text_data, extract_text_data_easyOCR,
    highlight_phrases_in_rectangle, highlight_phrases_outside_rectangles, extract_relation
)
from validation import (
    check_use_case_labels, validate_stickmen, check_arrow_near_labels,
    is_arrow_broken, relationships, isUseCasesOval
)
from helpers import is_intersecting

def main():
    STICKMAN_MODEL_PATH = 'Models/Stickman.pt'
    OVAL_MODEL_PATH = 'Models/OvalModel.pt'
    ARROW_MODEL_PATH = 'Models/ArrowModel.pt'
    ARROW_HEAD_MODEL_PATH = 'Models/ArrowHead.pt'
    LINE_MODEL_PATH = "Models/Line.pt"
    LINE_TO_OVAL_PATH = "Models/OvalToOval.pt"
    IMAGE_PATH = 'TestCases/2.png'
    LINE_THRESHOLD = 10
    VALID_COLOR = (0, 255, 0)
    INVALID_COLOR = (0, 0, 255)

    stickman_model = YOLO(STICKMAN_MODEL_PATH)
    oval_model = YOLO(OVAL_MODEL_PATH)
    arrow_model = YOLO(ARROW_MODEL_PATH)
    arrow_head_model = YOLO(ARROW_HEAD_MODEL_PATH)
    line_model = YOLO(LINE_MODEL_PATH)
    line_to_oval_model = YOLO(LINE_TO_OVAL_PATH)

    image = cv2.imread(IMAGE_PATH)
    height, width = image.shape[:2]
    image_area = height * width
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_image = cv2.Canny(blurred_image, 50, 150)
    contours, _ = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = cv2.HoughLinesP(canny_image, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    rectangles, rectangles_bbox = find_rectangles(lines, image)
    print(rectangles_bbox)
    phrase_bboxes = []
    outside_phrase_bboxes = []
    include_extends_bboxes = []
    text_bboxes = []
    if rectangles:
        for rect in rectangles:
            phrases, includes_extends, text = highlight_phrases_in_rectangle(rect, image)
            phrase_bboxes.extend(phrases)
            include_extends_bboxes.extend(includes_extends)
            text_bboxes.extend(text)
        outside_phrase_bboxes.extend(highlight_phrases_outside_rectangles(rectangles, image))

    ovals = detect_ovals(oval_model, image, rectangles)
    results = stickman_model(image)
    stickman_count, stickman_bboxes = process_stickmen_detections(results, rectangles, image)
    linesToOvals = detect_line_to_ovals(image, line_to_oval_model)
    arrows = detect_arrows(arrow_model, image)
    arrowHead = detect_arrow_head(image, arrow_head_model)

    is_arrow_broken(image, arrows, arrowHead)

    if rectangles:
        check_use_case_labels(ovals, image, phrase_bboxes, outside_phrase_bboxes)
        validate_stickmen(stickman_bboxes, outside_phrase_bboxes, ovals, image)

    isUseCasesOval(ovals, phrase_bboxes, image)

    if rectangles:
        check_arrow_near_labels(arrows, include_extends_bboxes, image)

    relationshipsArrow = relationships(arrows, ovals, arrowHead, text_bboxes)
    relations_map = {}
    for oval_id, related_ovals in relationshipsArrow.items():
        for related_oval, relation in related_ovals:
            relation_text = extract_relation(relation)
            if oval_id not in relations_map:
                relations_map[oval_id] = []
            relations_map[oval_id].append((related_oval, relation_text))

    for oval_id, related_ovals in relations_map.items():
        for related_oval, relation_text in related_ovals:
            if related_oval in relations_map:
                nested_relations = [r[1] for r in relations_map[related_oval]]
                if relation_text in nested_relations:
                    x1, y1, x2, y2 = ovals[related_oval]
                    text_x = x1
                    text_y = y1 - 10
                    cv2.putText(image, "'<<include>>', '<<extend>>' relationships should not be nested",  (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    config.NUMBER_OF_ERRORS += 1

    linesDetected, modifiedImage = detect_line(image, line_model)
    endpoints = get_line_endpoints(linesDetected)
    for i, (start, end) in enumerate(endpoints):
        if start and end:
            print(f"Line {i} - Start: {start}, End: {end}")
    does_line_intersect(modifiedImage, ovals, stickman_bboxes, linesDetected)

    x = int(width/2)
    if config.NUMBER_OF_ERRORS == 0:
        cv2.putText(image, "Use Case Diagram is Valid", (x,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

    max_height = 1080
    img_height, img_width = image.shape[:2]
    scaling_factor = max_height / img_height
    new_width = int(img_width * scaling_factor)
    new_height = max_height
    resized_image = cv2.resize(modifiedImage, (new_width, new_height))

    cv2.imshow("Use Case Diagram Checker", resized_image)
    cv2.imwrite("Output/outputImage.png", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

