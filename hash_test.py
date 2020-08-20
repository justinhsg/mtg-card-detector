import cv2
import numpy as np


def four_point_transform(image, quad):
    """
    A perspective transform for a quadrilateral polygon.
    Slightly modified version of the same function from
    https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector
    """
    
    # obtain a consistent order of the points and unpack them
    # individually
    rect = np.zeros((4, 2))
    (rect[:, 0], rect[:, 1]) = order_polygon_points(quad[:, 0], quad[:, 1])

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    # width_a = np.sqrt(((b_r[0] - b_l[0]) ** 2) + ((b_r[1] - b_l[1]) ** 2))
    # width_b = np.sqrt(((t_r[0] - t_l[0]) ** 2) + ((t_r[1] - t_l[1]) ** 2))
    width_a = np.sqrt(((rect[1, 0] - rect[0, 0]) ** 2) +
                      ((rect[1, 1] - rect[0, 1]) ** 2))
    width_b = np.sqrt(((rect[3, 0] - rect[2, 0]) ** 2) +
                      ((rect[3, 1] - rect[2, 1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(((rect[0, 0] - rect[3, 0]) ** 2) +
                       ((rect[0, 1] - rect[3, 1]) ** 2))
    height_b = np.sqrt(((rect[1, 0] - rect[2, 0]) ** 2) +
                       ((rect[1, 1] - rect[2, 1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order

    rect = np.array([
        [rect[0, 0], rect[0, 1]],
        [rect[1, 0], rect[1, 1]],
        [rect[2, 0], rect[2, 1]],
        [rect[3, 0], rect[3, 1]]], dtype="float32")

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    transform = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform, (max_width, max_height))
    # return the warped image
    return warped
    
def order_polygon_points(x, y):
    """
    Orders polygon points into a counterclockwise order.
    x_p, y_p are the x and y coordinates of the polygon points.
    """
    angle = np.arctan2(y - np.average(y), x - np.average(x))
    ind = np.argsort(angle)
    return (x[ind], y[ind])
    
    
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

def order_lines(lines):
    
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
    line_length = np.sqrt((line[1][1]-line[0][1])**2 + (line[1][0]-line[0][0])**2)
    return np.abs((line[1][1] - line[0][1]) * pt[0] - (line[1][0] - line[0][0]) * pt[1] + line[1][0]*line[0][1] - line[1][1]*line[0][0]) / line_length
    
def line_length(line):
    [[x1, y1], [x2,y2]] = line
    return np.sqrt((y2-y1)**2 + (x2-x1)**2)

def line_gradient(line):
    [[x1, y1], [x2,y2]] = line
    return np.arctan2(y2-y1, x2-x1)

def line_midpt(line):
    [[x1, y1], [x2,y2]] = line
    return np.array([(x1+x2)/2, (y1+y2)/2])
    
def process_lines(lines, image):

    four_lines = np.zeros((4, 2, 2))
    gradients = np.zeros(4)
    

    n_segments = 0
    segments = lines.reshape(len(lines), 2, 2)
    
    
    for segment in segments:        
        gradient = line_gradient(segment)
        mid_pt = line_midpt(segment)
        length = line_length(segment)
        to_add_line = True
        for i in range(n_segments):
            if(np.abs(gradients[i] - gradient) < 0.2):
                vert_dist = dist_to_line(four_lines[i], mid_pt)
                if(vert_dist < 10):
                    to_add_line = False
            
                    ori_length = line_length(four_lines[i])
                    
                    alpha = ori_length/(ori_length+length)
                    
                    four_lines[i] = alpha*four_lines[i] + (1-alpha)*segment
                    gradients[i] = line_gradient(four_lines[i])
                    
        if(to_add_line and n_segments < 4):
            
            four_lines[n_segments] += segment
            gradients[n_segments] += gradient
            n_segments += 1
            
    if n_segments !=4:
        return None
    else:
        return order_lines(four_lines)
    
    
    
def c3(card_contour, max_segment_area, image_height, image_width, image):
    image_area = image_height*image_width
    hull = cv2.convexHull(card_contour)
    
    hull_area = cv2.contourArea(hull)

    if (hull_area < 0.1 * max_segment_area or
            hull_area < image_area / 1000.):
        continue_segmentation = False
        is_card_candidate = False
        bound_quad = None
    elif (hull_area > 0.95*image_area):
        continue_segmentation = True
        is_card_candidate = False
        bound_quad = None
    else:
        continue_segmentation = True
        
        mask = np.zeros((image_height,image_width,1),np.uint8)
        cv2.drawContours(mask, [hull], 0, 255)
        
        linesP = cv2.HoughLinesP(mask, 1, np.pi/180, 25, None, 30 , 300)
        
        coords = np.zeros((4,2), np.int32)
        
        if linesP is not None and len(linesP)>=4:
            
            
            sorted_lines = process_lines(linesP[:, 0], image)
            if(sorted_lines is not None):
                for i in range(0, 4):
            
                    l1 = sorted_lines[i]
                    l2 = sorted_lines[(i+1)%4]
                    
                    intersect = line_intersection(l1, l2)
                    
                    if(intersect is None):
                        bound_quad = None
                        is_card_candidate = False
                        break

                    else:
                        coords[i]=(np.int32(np.asarray(intersect)))
                        quad_area = cv2.contourArea(coords)
                        is_card_candidate = 0.1 * max_segment_area < quad_area < image_area * 0.8
                        bound_quad = coords
            else:
                bound_quad = None
                is_card_candidate = False

        else:
            bound_quad = None
            is_card_candidate = False
        
        
        
    return (continue_segmentation,
            is_card_candidate,
            bound_quad)
    
def characterize_card_contour(card_contour,
                              max_segment_area,
                              image_height,
                              image_width):

                              
    image_area = image_height*image_width
    hull = cv2.convexHull(card_contour)
    hull_area = cv2.contourArea(hull)

    if (hull_area < 0.1 * max_segment_area or
            hull_area < image_area / 1000.):
        continue_segmentation = False
        is_card_candidate = False
        bound_quad = None
        
    else:
        continue_segmentation = True
        
        mask = np.zeros((image_height,image_width,1),np.uint8)
        cv2.drawContours(mask, [hull], 0, 255)
        
        linesP = cv2.HoughLinesP(mask, 1, np.pi/180, 25, None, 30 , 300)
        
        coords = np.zeros((4,2), np.int32)
        
        if linesP is not None and len(linesP)>=4:
        
            process_lines(linesP)
            sorted_lines = order_lines(linesP[:4,0])
            
            for i in range(0, 4):
        
                l1 = sorted_lines[i]
                l2 = sorted_lines[(i+1)%4]
                intersect = line_intersection((l1[0],l1[2], l2[0], l2[2]), (l1[1],l1[3], l2[1], l2[3]))
                
                if(intersect == (np.nan, np.nan)):
                    bound_quad = None
                    is_card_candidate = False
                    break

                else:
                    coords[i]=(np.int32(np.asarray(intersect)))
                    quad_area = cv2.contourArea(coords)
                    
                    
                    is_card_candidate = 0.1 * max_segment_area < quad_area < image_area * 0.8
                    bound_quad = coords

        else:
            bound_quad = None
            is_card_candidate = False
        
        
        
    return (continue_segmentation,
            is_card_candidate,
            bound_quad)
            
def contour_image(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    fltr_size = 1 + 2 * (min(im.shape[0], im.shape[1]) // 20)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, fltr_size, 10)
    contours, _ = cv2.findContours( np.uint8(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours_sorted
    
def applyCLAHE(im, clahe):
    lab=cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    lab[...,0] = clahe.apply(lab[...,0])
    im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return im

def phash(im):
    import scipy.fftpack
    imadj = cv2.resize(im, (32*4, 32*4), cv2.INTER_AREA)
    res=scipy.fftpack.dct(scipy.fftpack.dct(imadj, axis=0), axis=1)
    lowfreq = res[:32, :32]
    median = np.median(lowfreq, axis=(0,1))
    result = np.array(list(map(lambda x: x.flatten(), np.split(lowfreq > median, 3, axis = 2)))).flatten()
    return result
    
    
def process_image(image, name_to_idx, names, hashes):
    im = image.copy()
    maxsize = 1000
    if min(im.shape[0], im.shape[1]) > maxsize:
        scalef = maxsize/min(im.shape[0], im.shape[1])
        im = cv2.resize(im, ( int(im.shape[1] * scalef), int(im.shape[0] * scalef) ), interpolation=cv2.INTER_AREA)
    
    
    im_area = im.shape[0] * im.shape[1]
    max_segment_area = 0.01
    contours = contour_image(im)
    candidates = []
    
    lab=cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    lab[...,0] = clahe.apply(lab[...,0])
    im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    
    idx = 0
    for card_contour in contours:
        h,w,_ = im.shape
        (continue_segmentation,
            is_card_candidate,
            bounding_quad) = c3(card_contour, max_segment_area, h,w, image)
           
        if not continue_segmentation:
            break
        if is_card_candidate:
            
            if max_segment_area < 0.1:
                max_segment_area = cv2.contourArea(bounding_quad)
            warped = four_point_transform(image, bounding_quad)
            h,w,_ = warped.shape
            
            if(1.3 < h/w < 1.5):
                candidates.append((warped, bounding_quad))
            elif(1.3 < w/h < 1.5):
                candidates.append((cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE), bounding_quad))
                              
    matched = []
    
    for i, candidate in enumerate(candidates):
        (warped, bounding_quad) = candidate
        
        recog_thresh = 4
        
        card_name = 'unknown'
        is_recognised = False
        recognition_score = 0.

        for j in range(2):
            if j==1:
                phash_im = phash(cv2.rotate(warped, cv2.ROTATE_180))
            else:
                phash_im = phash(warped)
            dists = np.count_nonzero(phash_im != hashes, axis=1)
            
            min_arg = np.argmin(dists, axis = 0)
            min_val = dists[min_arg]
            
            duplicate_args = np.where(names == names[min_arg])
            
            dist_without = np.delete(dists, duplicate_args)
            avg = np.average(dist_without)
            std = np.std(dist_without)
            diff_measure = (avg-min_val)/std
            print(f"diff_measure: {diff_measure}, card: {names[min_arg]}")
            if(diff_measure >= recog_thresh and diff_measure >= recognition_score):
                recognition_score = diff_measure
                card_name = names[min_arg]
                is_recognised = True
        if(is_recognised):
            matched.append([card_name, recognition_score, False, bounding_quad])
    
    for i, match in enumerate(matched):
        [name, score, is_fragment, quad] = match
        min_coords = np.amin(quad, 0)
        max_coords = np.amax(quad, 0)
        if(is_fragment):
            continue
        for other in matched[i+1:]:
            [o_name, o_score, o_is_fragment, o_quad] = other
            if(o_is_fragment):
                continue
            if((min_coords <= o_quad).all() and (o_quad <= max_coords).all()):
                if(score >= o_score or name == o_name):
                    other[2] = True
                else:
                    match[2] = True
                    break
        if(match[2]):
            continue
            
        
        
        pts = quad.reshape((-1,1,2))
        middle = (np.average(quad[:, 0]), np.average(quad[:, 1]))
        fontScale = 0.5
        fontColor = (255,255,255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        (size, baseline) = cv2.getTextSize(name, font, fontScale, thickness)
        
        origin = ((int) (middle[0]-size[0]/2), (int) (middle[1]+size[1]/2))
        topleft = ((int) (middle[0]-size[0]/2), (int) (middle[1]-size[1]/2))
        bottomright = ((int) (middle[0]+size[0]/2), (int) (middle[1]+size[1]/2))
        cv2.polylines(image, np.int32([pts]), True, (0,255,0), thickness=2)
        cv2.rectangle(image, topleft, bottomright, (0,0,0), -1)
        cv2.putText(image, name, origin, font, fontScale,fontColor, thickness=thickness)
    return 
    
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


if __name__ == "__main__":
    from sys import argv
    
    
    from create_cache import load_cache
    
    cache = load_cache("TK")
    
    if(cache == None):
        print("Failed to load cache", file = sys.stderr)
        sys.exit(1)
    
    name_to_idx, names, hashes = cache

    '''
    filename = argv[1]
    ori = cv2.imread(filename)
    process_image(ori,name_to_idx, names , hashes)
        
    
    cv2.imshow("Contours", ori)
    key = cv2.waitKey(0)
    if(key == 27):
        cv2.destroyWindow("Contours")
    
    '''
    vc = cv2.VideoCapture(1)
    
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        rval, frame = vc.read()
        process_image(frame,name_to_idx, names, hashes)
        cv2.imshow("preview", frame)
        key = cv2.waitKey(1000)
        #break
        if key == 27: # exit on ESC
            break
    vc.release()
    cv2.destroyWindow("preview")
    #'''