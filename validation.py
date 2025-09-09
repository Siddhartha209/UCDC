import cv2
import numpy as np
from text_utils import extract_text_data
from nltk import pos_tag, word_tokenize
import config

def check_use_case_labels(ovals, image, phrase_bboxes, outside_phrase_bboxes):
    """Check that each oval has a valid label inside or outside its bounding box."""
    all_labelled = True
    requirement_6_met = False
    for oval in ovals:
        x1, y1, x2, y2 = oval
        labelled = False
        for bbox in phrase_bboxes:
            phrase_x_min, phrase_y_min, phrase_x_max, phrase_y_max = bbox
            if x1 <= phrase_x_min and y1 <= phrase_y_min and x2 >= phrase_x_max and y2 >= phrase_y_max:
                text_data = extract_text_data(image[phrase_y_min:phrase_y_max, phrase_x_min:phrase_x_max])
                config.VALID_LABELS += 1
                labelled = True
                for item in text_data:
                    phrase_text = item[1].strip()
                    phrase_pos = pos_tag(word_tokenize(phrase_text))
                    verbs = [word for word, pos in phrase_pos if pos in {'VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ'}]
                    nouns = [word for word, pos in phrase_pos if pos in {'NN', 'NNS', 'NNP', 'NNPS'}]
                    if verbs or nouns:
                        requirement_6_met = True
                break
        if not labelled:
            for bbox in outside_phrase_bboxes:
                phrase_x_min, phrase_y_min, phrase_x_max, phrase_y_max = bbox
                if x1 <= phrase_x_min and y1 <= phrase_y_min and x2 >= phrase_x_max and y2 >= phrase_y_max:
                    text_data = extract_text_data(image[phrase_y_min:phrase_y_max, phrase_x_min:phrase_x_max])
                    config.VALID_LABELS += 1
                    labelled = True
                    for item in text_data:
                        phrase_text = item[1].strip()
                        phrase_pos = pos_tag(word_tokenize(phrase_text))
                        verbs = [word for word, pos in phrase_pos if pos in {'VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ'}]
                        nouns = [word for word, pos in phrase_pos if pos in {'NN', 'NNS', 'NNP', 'NNPS'}]
                        if verbs or nouns:
                            requirement_6_met = True
                    break
        if not labelled:
            all_labelled = False
            cv2.putText(image, "Unlabelled use case", (x1, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            config.NUMBER_OF_ERRORS += 1

def validate_stickmen(stickman_bboxes, phrase_bboxes, ovals, image):
    """Check that each stickman has a valid actor label underneath and not inside an oval."""
    assigned_phrases = set()
    assigned_stickmen = set()
    phrases_to_ignore = set()
    for oval in ovals:
        x1, y1, x2, y2 = oval
        width = x2 - x1
        height = y2 - y1
        enlarged_x1 = int(x1 - width * 0.1)
        enlarged_y1 = int(y1 - height * 0.1)
        enlarged_x2 = int(x2 + width * 0.1)
        enlarged_y2 = int(y2 + height * 0.1)
        for i, phrase_bbox in enumerate(phrase_bboxes):
            px_min, py_min, px_max, py_max = phrase_bbox
            if not (px_max < enlarged_x1 or px_min > enlarged_x2 or py_max < enlarged_y1 or py_min > enlarged_y2):
                phrases_to_ignore.add(i)
    for i, stickman_bbox in enumerate(stickman_bboxes):
        sx_min, sy_min, sx_max, sy_max = stickman_bbox
        best_match = None
        bbox_height = sy_max - sy_min
        extended_sy_max = sy_max + int(0.5 * bbox_height)
        for j, phrase_bbox in enumerate(phrase_bboxes):
            if j in assigned_phrases or j in phrases_to_ignore:
                continue
            px_min, py_min, px_max, py_max = phrase_bbox
            top_center_x = (px_min + px_max) // 2
            top_center_y = py_min
            if sx_min <= top_center_x <= sx_max and sy_min <= top_center_y <= extended_sy_max:
                best_match = j
                break
        if best_match is not None:
            assigned_phrases.add(best_match)
            assigned_stickmen.add(i)
        else:
            cv2.putText(image, "Unlabelled Actor", (sx_min, sy_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            config.NUMBER_OF_ERRORS += 1
    incorrect_actors = False
    for i, phrase_bbox in enumerate(phrase_bboxes):
        if i not in assigned_phrases and i not in phrases_to_ignore:
            incorrect_actors = True
            px_min, py_min, px_max, py_max = phrase_bbox
            cv2.putText(image, "Incorrect Actor Detected", (px_min, py_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            config.NUMBER_OF_ERRORS += 1

def check_arrow_near_labels(arrows, labels, image):
    """Check that each arrow is near a label and each label is near an arrow."""
    for arrow in arrows:
        ax1, ay1, ax2, ay2 = arrow
        width = ax2 - ax1
        height = ay2 - ay1
        expansion_x = width * 0.25
        expansion_y = height * 0.25
        enlarged_ax1 = int(ax1 - expansion_x)
        enlarged_ay1 = int(ay1 - expansion_y)
        enlarged_ax2 = int(ax2 + expansion_x)
        enlarged_ay2 = int(ay2 + expansion_y)
        arrow_has_label = False
        for label in labels:
            lx1, ly1, lx2, ly2 = label
            if not (lx2 < enlarged_ax1 or lx1 > enlarged_ax2 or ly2 < enlarged_ay1 or ly1 > enlarged_ay2):
                arrow_has_label = True
                break
        if not arrow_has_label:
            cv2.putText(image, "Links between usecases must be labelled '<<include>>' or '<<extend>>'",
                       (ax1, ay1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            config.NUMBER_OF_ERRORS += 1
    for label in labels:
        lx1, ly1, lx2, ly2 = label
        width = lx2 - lx1
        height = ly2 - ly1
        expansion_x = width * 0.25
        expansion_y = height * 0.25
        enlarged_lx1 = int(lx1 - expansion_x)
        enlarged_ly1 = int(ly1 - expansion_y)
        enlarged_lx2 = int(lx2 + expansion_x)
        enlarged_ly2 = int(ly2 + expansion_y)
        label_has_arrow = False
        for arrow in arrows:
            ax1, ay1, ax2, ay2 = arrow
            if not (ax2 < enlarged_lx1 or ax1 > enlarged_lx2 or ay2 < enlarged_ly1 or ay1 > enlarged_ly2):
                label_has_arrow = True
                break
        if not label_has_arrow:
            cv2.putText(image, "'<<Include>>, '<<Extend>>' must be represented as broken arrows",
                       (lx1, ly1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            config.NUMBER_OF_ERRORS += 1

def is_arrow_broken(image, arrow_boxes, arrowhead_boxes):
    """Check that arrow heads are not connected to arrow bodies (broken arrows)."""
    def do_boxes_intersect(box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        if x2_1 < x1_2 or x2_2 < x1_1:
            return False
        if y2_1 < y1_2 or y2_2 < y1_1:
            return False
        return True
    for arrowhead_box in arrowhead_boxes:
        is_connected = False
        for arrow_box in arrow_boxes:
            if do_boxes_intersect(arrowhead_box, arrow_box):
                is_connected = True
                break
        if not is_connected:
            x1, y1, x2, y2 = arrowhead_box
            text_position = (x1, y1 - 10)
            cv2.putText(image, "Links between use cases must be broken", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def relationships(arrows, ovals, arrowHead, text_bboxes):
    """Link ovals based on overlap with arrows and arrow heads, and associate relationship text."""
    relationships_dict = {}
    enlarged_ovals = []
    for oval_id, (x1, y1, x2, y2) in enumerate(ovals):
        width, height = x2 - x1, y2 - y1
        enlarged_x1, enlarged_y1 = int(x1 - width * 0.125), int(y1 - height * 0.125)
        enlarged_x2, enlarged_y2 = int(x2 + width * 0.125), int(y2 + height * 0.125)
        enlarged_ovals.append((oval_id, enlarged_x1, enlarged_y1, enlarged_x2, enlarged_y2))
    def is_intersecting(box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        return not (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2)
    def is_box_overlapping(arrow_head_box, enlarged_box):
        return is_intersecting(arrow_head_box, enlarged_box)
    for arrow_idx, arrow in enumerate(arrows):
        arrow_x1, arrow_y1, arrow_x2, arrow_y2 = arrow
        if arrow_idx < len(arrowHead):
            arrow_head_box = arrowHead[arrow_idx]
        else:
            continue
        overlapping_ovals = []
        for oval_id, enlarged_x1, enlarged_y1, enlarged_x2, enlarged_y2 in enlarged_ovals:
            enlarged_box = (enlarged_x1, enlarged_y1, enlarged_x2, enlarged_y2)
            if is_intersecting(arrow, enlarged_box):
                overlapping_ovals.append(oval_id)
        if len(overlapping_ovals) == 2:
            oval1_id, oval2_id = overlapping_ovals
            destination_oval = None
            for oval_id, enlarged_x1, enlarged_y1, enlarged_x2, enlarged_y2 in enlarged_ovals:
                enlarged_box = (enlarged_x1, enlarged_y1, enlarged_x2, enlarged_y2)
                if is_box_overlapping(arrow_head_box, enlarged_box) and oval_id in overlapping_ovals:
                    destination_oval = oval_id
                    break
            closest_text = None
            min_distance = float('inf')
            arrow_center_x = (arrow_x1 + arrow_x2) / 2
            arrow_center_y = (arrow_y1 + arrow_y2) / 2
            for text_box, phrase_text in text_bboxes:
                text_x1, text_y1, text_x2, text_y2 = text_box
                text_center_x = (text_x1 + text_x2) / 2
                text_center_y = (text_y1 + text_y2) / 2
                distance = ((arrow_center_x - text_center_x) ** 2 + (arrow_center_y - text_center_y) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    closest_text = phrase_text
            max_distance_threshold = 100
            if min_distance > max_distance_threshold:
                closest_text = ""
            if destination_oval is not None:
                if destination_oval == oval1_id:
                    source_oval = oval2_id
                    dest_oval = oval1_id
                else:
                    source_oval = oval1_id
                    dest_oval = oval2_id
                if source_oval not in relationships_dict:
                    relationships_dict[source_oval] = []
                if not any(dest == dest_oval for dest, _ in relationships_dict[source_oval]):
                    relationships_dict[source_oval].append((dest_oval, closest_text))
    return relationships_dict

def isUseCasesOval(ovals, phrase_bboxes, image):
    """Check that use case labels are inside ovals."""
    counter = 0
    for phrase_bbox in phrase_bboxes:
        phrase_x1, phrase_y1, phrase_x2, phrase_y2 = phrase_bbox
        inside_oval = False
        for oval in ovals:
            oval_x1, oval_y1, oval_x2, oval_y2 = oval
            if (oval_x1 <= phrase_x1 <= oval_x2 and oval_y1 <= phrase_y1 <= oval_y2 and
                oval_x1 <= phrase_x2 <= oval_x2 and oval_y1 <= phrase_y2 <= oval_y2):
                inside_oval = True
                break
        if not inside_oval and config.IS_BOUNDING_BOX_LABELLED == False:
            cv2.putText(image, "Use cases must be ovals", (phrase_x1, phrase_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            config.NUMBER_OF_ERRORS += 1
        elif not inside_oval:
            counter += 1
            if counter > 1:
                cv2.putText(image, "Use cases must be ovals", (phrase_x1, phrase_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                config.NUMBER_OF_ERRORS += 1
