import cv2
import numpy as np

def line_intersection(line1, line2):
    """
    Intersection point of 2 line segments (x1, y1) and (x2, y2) , first express each line as a function
    Ax + By = C, where 
    A = y2-y1
    B = -(x2-x1)
    C = Ax1 + By2
    
    Then, use Cramer's rule to solve.
    """
    lines = np.array([line1, line2])
    B_vect, A_vect = (lines[:,1] - lines[:,0]).T
    B_vect = -B_vect
    C_vect = np.diagonal(lines[:,0] @ [A_vect, B_vect])
    det_AB = np.linalg.det([A_vect, B_vect])
    if(det_AB == 0):
        return None
    else:
        det_CB = np.linalg.det([C_vect, B_vect])
        det_AC = np.linalg.det([A_vect, C_vect])
        return np.array([det_CB, det_AC])/det_AB

def order_points(points):
    """
    Orders polygon points into a clockwise order, starting from the top left.
    """
    angle = np.arctan2(points[:,1] - np.average(points[:,1]), points[:,0] - np.average(points[:,0]))
    ind = np.argsort(angle)
    return points[ind]
    
        
def order_lines(lines):
    """
        Orders lines segments in a counter-clockwise order, based on their midpoints
    """
    midpts = np.zeros((4,2),np.int32)
    for i, line in enumerate(lines):
        midpts[i] = (line[0,:]+line[1,:])/2
    cy, cx = np.average(midpts[:,0]), np.average(midpts[:,1])
    angle = np.arctan2(midpts[:,0] - cy, midpts[:,1] - cx)
    ind = np.argsort(angle)
    return lines[ind]
    
def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled
    
    
def dist_to_line(line, pt):
    length = line_length(line)
    return np.abs((line[1][1] - line[0][1]) * pt[0] 
                   -(line[1][0] - line[0][0]) * pt[1]
                   + line[1][0]*line[0][1]
                   - line[1][1]*line[0][0]) / length
    
def line_length(line):
    [[x1, y1], [x2,y2]] = np.int32(line)
    return np.sqrt((y2-y1)**2 + (x2-x1)**2)

def line_gradient(line):
    [[x1, y1], [x2,y2]] = line
    return np.arctan2(y2-y1, x2-x1)

def line_midpt(line):
    [[x1, y1], [x2,y2]] = line
    return np.array([(x1+x2)/2, (y1+y2)/2])