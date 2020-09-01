import cv2
import numpy as np
import sys

from helper_geometry import line_gradient, line_midpt, line_length, dist_to_line, order_lines, line_intersection, scale_contour
from helper_cv2 import four_point_transform, applyCLAHE, phash, apply_offset, apply_trim, add_text_bg

class CardIdentifier:
    
    max_card_size = 0.9
    min_card_size = 0.01
    iden_threshold = 3.5
    
    
    def __init__(self, cache):
        self.cached_name_to_idx, self.cached_names, self.cached_hashes = cache
        self.offsets = np.int8([0,0,0])
        self.trim = 1.
        
    def set_offsets(self, new_offsets):
        if new_offsets is not None:
            self.offsets = new_offsets 
 
    def set_trim(self, new_trim):
        if new_trim is not None:
            self.trim = new_trim
    
    def get_card_candidates(self, image):
        image_copy = applyCLAHE(image)
        image_height, image_width, _ = image.shape
        image_area = image_height*image_width
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fltr_size = 1 + 2 * (min(image.shape[0], image.shape[1]) // 20)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, fltr_size, 10)
        contours, _ = cv2.findContours( np.uint8(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours= list(filter(lambda x: self.min_card_size < cv2.contourArea(x)/image_area < self.max_card_size, contours))
        sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
        
        
        
        card_candidates = []
        for contour in sorted_contours:
            (continue_search, 
             is_candidate, 
             quad) = self.get_candidate(contour, image_height, image_width)
        
            if not continue_search:
                break
            if is_candidate:
                transformed_card = four_point_transform(image, quad)
                card_height, card_width, _ = transformed_card.shape
                if(1.3 < card_height/card_width < 1.5):
                    card_candidates.append((transformed_card, quad))
                if(1.3 < card_width/ card_height < 1.3):
                    card_candidates.append(cv2.rotate(transformed_card, cv2.ROTATE_90_CLOCKWISE), quad)
        return card_candidates
                
    def get_candidate(self, contour, image_height, image_width):
        image_area = image_height*image_width
        convex_hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(convex_hull)
        
        continue_search = False
        is_card_candidate = False
        quad = np.zeros((4,2), np.int32)
        
        
        if(hull_area > image_area*self.min_card_size):
            continue_search = True
            
            if (hull_area < image_area*self.max_card_size):
            
                mask = np.zeros((image_height,image_width,1),np.uint8)
                cv2.drawContours(mask, [convex_hull], 0, 255)
                linesP = cv2.HoughLinesP(mask, 1, np.pi/180, 25, None, 30 , 300)
            
                sorted_lines = self.process_lines(linesP[:, 0])
                if(sorted_lines is not None):
                    for i in range(0,4):
                        line1 = sorted_lines[i]
                        line2 = sorted_lines[(i+1)%4]
                        
                        intersect = line_intersection(line1,line2)
                        
                        if(intersect is None):
                            break
                        else:
                            quad[i] = np.int32(intersect)
                    is_card_candidate = self.min_card_size*image_area < cv2.contourArea(quad) < self.max_card_size*image_area
        return (continue_search, is_card_candidate, quad)
        
    def process_lines(self, lines):
        four_lines = np.zeros((4,2,2))
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
            
            
    def identify_candidates(self, candidate_list):
        
        identified = []
        
        for candidate in candidate_list:
            (cand_img, cand_quad) = candidate
            
            card_name = 'unknown'
            is_recognised = False
            best_score = 0.
            
            
            
            variants = [cand_img, cv2.rotate(cand_img, cv2.ROTATE_180)]
            for i in range(len(variants)):
                variants.append(apply_offset(variants[i], self.offsets))
            for i in range(len(variants)):
                variants.append(apply_trim(variants[i], self.trim))
            for test_img in variants:
            
                #test_offset = apply_offset(test_img, self.offsets)
                #test_trim = apply_trim(test_offset, self.trim)
            
                test_phash = phash(test_img)
                
                hash_dists = np.count_nonzero(test_phash != self.cached_hashes, axis = 1)
                
                min_dist = np.min(hash_dists, axis=0)
                poss_name = self.cached_names[np.argmin(hash_dists, axis=0)]
                #print(self.cached_name_to_idx["Teysa Karlov"], hash_dists[113])
                
                other_dists = np.delete(hash_dists, self.cached_name_to_idx[poss_name])
                avg = np.average(other_dists)
                std = np.std(other_dists)
                diff_measure = (avg-min_dist)/std
                #print(f"{poss_name}: {diff_measure}")
                if(diff_measure >= self.iden_threshold and diff_measure >= best_score):
                    best_score = diff_measure
                    card_name = poss_name
                    is_recognised = True
            
            if(is_recognised):
                identified.append({"name"  : card_name,
                                   "score" : best_score,
                                   "quad"  : cand_quad, 
                                   "centre":np.int32(np.average(cand_quad, axis=0)),
                                   "is_dupe": False})
                
        
        unique_identified = []
        for i, result in enumerate(identified):
            
            if(result["is_dupe"]):
                continue
            min_coords = np.amin(result["quad"], axis=0)
            max_coords = np.amax(result["quad"], axis=0)
            for other in identified[i+1:]:
                if(other["is_dupe"]):
                    continue
                if ( (min_coords <= other["quad"]).all() and 
                     (max_coords >= other["quad"]).all() ):
                    if(result["score"] >= other["score"] or result["name"] == other["name"]):
                        other["is_dupe"] = True
                    else:
                        other["is_dupe"] = True
                        break
            if(not result["is_dupe"]):
                unique_identified.append(result)
        return unique_identified
    
    
    def display_results(self, image, results):
        for result in results:
            cv2.polylines(image, [result["quad"]], True, (255,255,0))
            add_text_bg(image, result["name"], result["centre"])

if __name__ == "__main__":
    import argparse
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_code", help="code of the cache to use")
    parser.add_argument("-c", "--calibrate", dest="fuzzy_name", help="card name to calibrate")
    parser.add_argument("-a", "--auto", action="store_true", help="automatically detect card to calibrate")
    args = parser.parse_args()
    
    
    from create_cache import load_cache, cache_exists
    
    if(not cache_exists(args.cache_code)):
        print("Cache does not exist", file=sys.stderr)
        exit(1)
    
    cache = load_cache(args.cache_code)
    identifier = CardIdentifier(cache)
    
    
    if(args.fuzzy_name is not None):
        from calibration import Calibrator
        calibrator = Calibrator(cache, args.fuzzy_name, args.auto)
        calib_offset, calib_trim = calibrator.calibrate()
        identifier.set_offsets(calib_offset)
        identifier.set_trim(calib_trim)
        
        
    
    
    
    vc = cv2.VideoCapture(1)
    
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        rval, frame = vc.read()
        
        
        candidates = identifier.get_card_candidates(frame)
        results = identifier.identify_candidates(candidates)
        identifier.display_results(frame, results)
        cv2.imshow("preview", frame)
        key = cv2.waitKey(150)
        if key == 27: # exit on ESC
            break
    vc.release()
    cv2.destroyWindow("preview")
    cv2.destroyAllWindows()