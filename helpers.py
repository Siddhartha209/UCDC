def is_intersecting(rect, stickman_box):
    """Function to check if a bounding box (stickman_box) intersects with a rectangle (rect)"""
    x1, y1, x2, y2 = stickman_box
    rect_x1, rect_y1 = rect[0][0][0], rect[0][0][1]
    rect_x2, rect_y2 = rect[2][0][0], rect[2][0][1]
    return not (x2 <= rect_x1 or x1 >= rect_x2 or y2 <= rect_y1 or y1 >= rect_y2)