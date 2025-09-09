import pytesseract
import cv2
import numpy as np
import re
import easyocr
import nltk
import config
from nltk import pos_tag, word_tokenize

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
reader = easyocr.Reader(['en'])
LINE_THRESHOLD = 10
VALID_COLOR = (0, 255, 0)
INVALID_COLOR = (0, 0, 255)

def is_valid_word(word, pos):
    """Check if a word is a valid verb or noun for use case labeling."""
    active_verbs = {'VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ'}
    nouns = {'NN', 'NNS', 'NNP', 'NNPS'}
    return pos in active_verbs or pos in nouns

def extract_text_data(roi):
    """Extract text data from an image region using pytesseract."""
    return pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT)

def extract_text_data_easyOCR(roi):
    """Extract text data from an image region using EasyOCR."""
    results = reader.readtext(roi, detail=1)
    return results

def highlight_phrases_in_rectangle(rect, image):
    """Highlight and validate phrases inside a rectangle (use case box)."""
    x1, y1, x2, y2 = rect[0][0][0], rect[0][0][1], rect[2][0][0], rect[2][0][1]
    top_box_height = int((y2 - y1) * 0.1)
    roi = image[y1:y2, x1:x2]
    text_data = extract_text_data_easyOCR(roi)

    words = [{'word': item[1].strip(),
              'x': item[0][0][0],
              'y': item[0][0][1],
              'width': item[0][2][0] - item[0][0][0],
              'height': item[0][2][1] - item[0][0][1]}
             for item in text_data if item[1].strip()]
    words.sort(key=lambda w: w['y'])

    phrases, current_phrase, last_y = [], [], -1
    phrase_bboxes = []
    includes_extends_bboxes = []
    text_bboxes = []
    seen_phrases = set()

    for word in words:
        if current_phrase and abs(word['y'] - last_y) > LINE_THRESHOLD:
            phrases.append(current_phrase)
            current_phrase = []
        current_phrase.append(word)
        last_y = word['y']
    if current_phrase:
        phrases.append(current_phrase)

    phrases_in_top_box = False

    for phrase in phrases:
        phrase_text = " ".join([w['word'] for w in phrase])
        phrase_pos = pos_tag(word_tokenize(phrase_text))
        is_action_phrase = any(is_valid_word(word, pos) for word, pos in phrase_pos)
        config.DETECTED_LABELS += 1

        if is_action_phrase:
            config.IS_VERB_LABELS += 1

        color = VALID_COLOR if is_action_phrase else INVALID_COLOR
        x_min = int(min(w['x'] for w in phrase) + x1)
        y_min = int(min(w['y'] for w in phrase) + y1)
        x_max = int(max(w['x'] + w['width'] for w in phrase) + x1)
        y_max = int(max(w['y'] + w['height'] for w in phrase) + y1)

        if y_min < y1 + top_box_height:
            phrases_in_top_box = True

        if any(keyword in phrase_text.lower() for keyword in ["include", "includes", "extend", "extends", "<<include>>", "<<includes>>", "<<extend>>", "<<extends>>"]):
            config.INCLUDE_EXTENDS_COUNTER += 1
            includes_extends_bboxes.append((x_min, y_min, x_max, y_max))
            text_bboxes.append(((x_min, y_min, x_max, y_max), phrase_text))
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), VALID_COLOR, 2)
        else:
            if phrase_text.lower() in seen_phrases:
                cv2.putText(image, "Duplicate Use Case Detected", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INVALID_COLOR, 2)
                config.NUMBER_OF_ERRORS += 1

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            if color == INVALID_COLOR:
                cv2.putText(image, "Label must be written as verb or noun", (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, INVALID_COLOR, 2)
                config.NUMBER_OF_ERRORS += 1

            print(f"Phrase: '{phrase_text}' Bounding Box: ({x_min}, {y_min}, {x_max}, {y_max})")
            phrase_bboxes.append((x_min, y_min, x_max, y_max))
        seen_phrases.add(phrase_text.lower())

    if not phrases_in_top_box:
        cv2.putText(image, "Bounding Box must be labelled", (x1 + 10, y1 + int(top_box_height/2)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, INVALID_COLOR, 2)
        config.NUMBER_OF_ERRORS += 1
        config.IS_BOUNDING_BOX_LABELLED = False
    return phrase_bboxes, includes_extends_bboxes, text_bboxes

def highlight_phrases_outside_rectangles(rectangles, image):
    """Highlight and validate phrases outside all rectangles (actor labels)."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for rect in rectangles:
        x1, y1 = rect[0][0]
        x2, y2 = rect[2][0]
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)

    inverted_mask = cv2.bitwise_not(mask)
    roi = cv2.bitwise_and(image, image, mask=inverted_mask)
    text_data = extract_text_data(roi)

    words = []
    for i in range(len(text_data['text'])):
        clean_word = ''.join(char for char in text_data['text'][i] if char.isalpha())
        if len(clean_word) >= 3:
            words.append({
                'word': text_data['text'][i].strip(),
                'clean_word': clean_word,
                'x': text_data['left'][i],
                'y': text_data['top'][i],
                'width': text_data['width'][i],
                'height': text_data['height'][i]
            })

    words.sort(key=lambda w: w['y'])
    phrases, current_phrase, last_y = [], [], -1

    for word in words:
        if current_phrase and abs(word['y'] - last_y) > LINE_THRESHOLD:
            phrases.append(current_phrase)
            current_phrase = []
        current_phrase.append(word)
        last_y = word['y']
    if current_phrase:
        phrases.append(current_phrase)

    phrase_bboxes = []
    for phrase in phrases:
        combined_clean_text = ''.join([w['clean_word'] for w in phrase])
        if len(combined_clean_text) < 3:
            continue

        x_min = min(w['x'] for w in phrase)
        y_min = min(w['y'] for w in phrase)
        x_max = max(w['x'] + w['width'] for w in phrase)
        y_max = max(w['y'] + w['height'] for w in phrase)
        phrase_bboxes.append((x_min, y_min, x_max, y_max))

        phrase_text = " ".join([w['word'] for w in phrase])
        phrase_pos = pos_tag(word_tokenize(phrase_text))
        config.ACTOR_LABELS += 1

        print(f"Phrase: '{phrase_text}' Bounding Box: ({x_min}, {y_min}, {x_max}, {y_max})")
    return phrase_bboxes

def extract_relation(relation):
    """Extract only alphabetic characters from a relation string."""
    return re.sub(r'[^a-zA-Z]', '', relation)