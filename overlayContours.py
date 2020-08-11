from copy import deepcopy
from itertools import product

from shapely.geometry import LineString
from shapely.geometry.polygon import Polygon
from shapely.affinity import scale
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob

def four_point_transform(image, poly):
    """
    A perspective transform for a quadrilateral polygon.
    Slightly modified version of the same function from
    https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector
    """
    pts = np.zeros((4, 2))
    pts[:, 0] = np.asarray(poly.exterior.coords)[:-1, 0]
    pts[:, 1] = np.asarray(poly.exterior.coords)[:-1, 1]
    # obtain a consistent order of the points and unpack them
    # individually
    rect = np.zeros((4, 2))
    (rect[:, 0], rect[:, 1]) = order_polygon_points(pts[:, 0], pts[:, 1])

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
    
def line_intersection(x, y):
    """
    Calculates the intersection point of two lines, defined by the points
    (x1, y1) and (x2, y2) (first line), and
    (x3, y3) and (x4, y4) (second line).
    If the lines are parallel, (nan, nan) is returned.
    """
    #print("line_inter: {} {}".format(x,y))
    slope_0 = (x[0] - x[1]) * (y[2] - y[3])
    slope_2 = (y[0] - y[1]) * (x[2] - x[3])
    if slope_0 == slope_2:
        # parallel lines
        xis = np.nan
        yis = np.nan
    else:
        xy_01 = x[0] * y[1] - y[0] * x[1]
        xy_23 = x[2] * y[3] - y[2] * x[3]
        denom = slope_0 - slope_2

        xis = (xy_01 * (x[2] - x[3]) - (x[0] - x[1]) * xy_23) / denom
        yis = (xy_01 * (y[2] - y[3]) - (y[0] - y[1]) * xy_23) / denom

    return (xis, yis)

def convex_hull_polygon(contour):
    """
    Returns the convex hull of the given contour as a polygon.
    """
    hull = cv2.convexHull(contour)
    phull = Polygon([[x, y] for (x, y) in
                     zip(hull[:, :, 0], hull[:, :, 1])])
    return phull
    
def order_lines(lines):

    midpts = np.zeros((4,2),np.int32)
    for i, line in enumerate(lines):
        midpts[i] = np.array([(line[0]+line[2])/2,(line[1]+line[3])/2])
    cy, cx = np.average(midpts[:,0]), np.average(midpts[:,1])
    angle = np.arctan2(midpts[:,0] - cy, midpts[:,1] - cx)
    ind = np.argsort(angle)
    return lines[ind]
    
    
            
def characterize_card_contour_(card_contour,
                              max_segment_area,
                              image_area,
                              im):
    """
    Calculates a bounding polygon for a contour, in addition
    to several charasteristic parameters.
    """
    phull = convex_hull_polygon(card_contour)
    if (phull.area < 0.1 * max_segment_area or
            phull.area < image_area / 1000.):
        # break after card size range has been explored
        continue_segmentation = False
        is_card_candidate = False
        bounding_poly = None
        crop_factor = 1.
    else:

        continue_segmentation = True
        h,w,_ = im.shape
        mask = np.zeros((h,w,1),np.uint8)
        
        hull = cv2.convexHull(card_contour)
        imc = im.copy()
        cv2.drawContours(imc, [hull], 0, (255,0,0))
        cv2.drawContours(mask, [hull], 0, 255)
        
        linesP = cv2.HoughLinesP(mask, 1, np.pi/180, 25, None, 50 , 300)
        coords = np.zeros((4,2), np.int32)
        if linesP is not None and len(linesP)>=4:
            sorted_lines = order_lines(linesP[:4,0])
            for i in range(0, 4):
                l1 = sorted_lines[i]
                l2 = sorted_lines[(i+1)%4]
                intersect = line_intersection((l1[0],l1[2], l2[0], l2[2]), (l1[1],l1[3], l2[1], l2[3]))
                
                if(intersect == (np.nan, np.nan)):
                    bounding_poly = None
                    is_card_candidate = False
                    crop_factor = None
                    break
                else:
                    coords[i]=(np.int32(np.asarray(intersect)))
                    bounding_poly = Polygon(coords)
                    is_card_candidate = 0.1 * max_segment_area < bounding_poly.area < image_area * 0.8
                    crop_factor = phull.area/bounding_poly.area if is_card_candidate else 0
        else:
            bounding_poly = None
            is_card_candidate = False
            crop_factor = None
        
        
        
        
    return (continue_segmentation,
            is_card_candidate,
            bounding_poly,
            crop_factor)
            
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
    imadj = cv2.resize(cv2.cvtColor(im , cv2.COLOR_BGR2GRAY), (32*4, 32*4), cv2.INTER_AREA)
    pixels = np.asarray(imadj)
    res = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
    lowfreq = res[:32, :32]
    med = np.median(lowfreq)
    diff = lowfreq>med
    return diff.flatten()
    
    
def process_image(image, name, phashes):
    im = image.copy()
    maxsize = 1000
    if min(im.shape[0], im.shape[1]) > maxsize:
        scalef = maxsize/min(im.shape[0], im.shape[1])
        im = cv2.resize(im, ( int(im.shape[1] * scalef), int(im.shape[0] * scalef) ), interpolation=cv2.INTER_AREA)
    lab=cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    lab[...,0] = clahe.apply(lab[...,0])
    im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    
    im_area = im.shape[0] * im.shape[1]
    max_segment_area = 0.01
    contours = contour_image(im)
    candidates = []
    idx = 0
    for card_contour in contours:
        try:    
            (continue_segmentation,
                 is_card_candidate,
                 bounding_poly,
                 crop_factor) = characterize_card_contour_(card_contour, max_segment_area, im_area, im)
        except NotImplementedError as nie:
            # this can occur in Shapely for some funny contour shapes
            # resolve by discarding the candidate
            (continue_segmentation,
             is_card_candidate,
             bounding_poly,
             crop_factor) = (True, False, None, 1.)
        if not continue_segmentation:
            break
        if is_card_candidate:
            if max_segment_area < 0.1:
                max_segment_area = bounding_poly.area
            warped = four_point_transform(image, scale(bounding_poly,
                                            xfact=0.95,
                                            yfact=0.95,
                                            origin='centroid'))
            h,w,_ = warped.shape
            if(1.3 < h/w < 1.5):
                candidates.append((warped,
                                  bounding_poly,
                                  bounding_poly.area / im_area))
            elif(1.3 < w/h < 1.5):
                candidates.append((rotate(warped,90),
                                  bounding_poly,
                                  bounding_poly.area / im_area))
            
    matched = []
    for i, candidate in enumerate(candidates):
        (warped, bounding_quad, _) = candidate
        #cv2.imshow("warped{}".format(i), warped)
        #cv2.waitKey(0)
        #bquad_corners = np.empty((4, 2))
        #bquad_corners[:, 0] = np.asarray(bounding_quad.exterior.coords)[:-1, 0]
        #bquad_corners[:, 1] = np.asarray(bounding_quad.exterior.coords)[:-1, 1]
        #pts = bquad_corners.reshape((-1,1,2))
        
        
        
        card_name = 'unknown'
        is_recognized = False
        recognition_score = 0.
        rotations = np.array([0., 180., ])
        d_0_dist = np.zeros(len(rotations))
        d_0 = np.zeros((len(phashes), len(rotations)))
        for j, rot in enumerate(rotations):
            if not -1.e-5 < rot < 1.e-5:
                phash_im = phash(rotate(warped, rot))
            else:
                phash_im = phash(warped)
            for i, ref_phash in enumerate(phashes):
                d_0[i,j] = np.count_nonzero(phash_im != ref_phash)
            
            d_0_ = d_0[d_0[:, j] > np.amin(d_0[:, j]), j]
            d_0_ave = np.average(d_0_)
            d_0_std = np.std(d_0_)
            d_0_dist[j] = (d_0_ave - np.amin(d_0[:, j])) / d_0_std
            if (d_0_dist[j] > 4 and np.argmax(d_0_dist) == j):
                card_name = names[np.argmin(d_0[:, j])]
                is_recognized = True
                recognition_score = d_0_dist[j] / 4
                matched.append([card_name, recognition_score, False, bounding_quad])
                break
    
    for match in matched:
        [name, score, is_fragment, quad] = match
        bquad_corners = np.empty((4, 2))
        bquad_corners[:, 0] = np.asarray(quad.exterior.coords)[:-1, 0]
        bquad_corners[:, 1] = np.asarray(quad.exterior.coords)[:-1, 1]
        pts = bquad_corners.reshape((-1,1,2))
        bounding_poly = Polygon([[x, y] for (x, y) in zip(bquad_corners[:, 0],  bquad_corners[:, 1])])
        
        
        
        middle = (np.average(bquad_corners[:, 0]), np.average(bquad_corners[:, 1]))
        
        
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
    
if __name__ == "__main__":
    from sys import argv
    
    
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    
    
    phashes = []
    names = []
    #'''
    filenames = glob.glob('.\\m21\\*.jpg')
    for filename in filenames:
        img = cv2.imread(filename)
        img_name = filename.split('.\\m21\\')[1]
        img = applyCLAHE(img, clahe)
        img_phash = phash(img)
        names.append(img_name)
        phashes.append(img_phash)
    phashes = np.array(phashes)
    #'''
    
    '''
    filename = argv[1]
    ori = cv2.imread(filename)
    process_image(ori,names, phashes)
        
    
    cv2.imshow("Contours", ori)
    key = cv2.waitKey(0)
    if(key == 27):
        cv2.destroyWindow("Contours")
    '''
    #'''
    vc = cv2.VideoCapture(1)
    
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        rval, frame = vc.read()
        process_image(frame,names, phashes)
        cv2.imshow("preview", frame)
        key = cv2.waitKey(200)
        #break
        if key == 27: # exit on ESC
            break
    vc.release()
    cv2.destroyWindow("preview")
    #'''