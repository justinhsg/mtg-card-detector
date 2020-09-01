from create_cache import load_cache, cache_exists, im_from_url
from helper_cv2 import four_point_transform, apply_offset, apply_trim, phash_dim, phash, add_text
from card_identifier import CardIdentifier
import cv2
import numpy as np
import argparse
import requests
import sys
class Calibrator:
    
    gallery_card_height = 200
    gallery_card_width = 140
    
    calib_height = 900
    
    def __init__(self, cache, fuzzy_name, search_auto):
    #def __init__(self, cache, card_name, search_uri, search_auto):
        self.cache_name_to_idx, self.cache_names, self.hashes = cache
        
        self.calib_name = None
        self.calib_img_idx = None
        
        self.search_auto = search_auto
        
        self.ref_imgs = []
        self.resized_imgs = []
        
        self.gallery_img = np.zeros((2*self.gallery_card_height, 4*self.gallery_card_width, 3), dtype=np.uint8)   
        self.gallery_page = None
        self.gallery_page_max = None
        
        
        self.cand_points = np.zeros((4,2), dtype=np.uint32)
        self.cand_n_points = 0
        self.cand_img = None
        
        self.show_cropped = False
        
        
        self.find_ref_full_name(fuzzy_name)
            
        
        
    def calibrate(self):
        if(len(self.ref_imgs)==0):
            print("No reference image")
            return (None, None)
        self.get_candidate()
        if(self.cand_img is None):
            print("No candidate image")
            return (None, None)
        return(self.calibrate_offsets(), self.calibrate_trim())
        
        
    def find_ref_full_name(self, fuzzy_name):
        r = requests.get(f"https://api.scryfall.com/cards/named?fuzzy={fuzzy_name}")
        response = r.json()
        if(response['object']!='card'):
            print(f"{response['details']}", file=sys.stderr)
            return
        else:
            self.calib_name = response['name']
            if(self.calib_name not in self.cache_names):
                print(f"{self.calib_name} is not in cache", file=sys.stderr)
                return
            search_uri = response['prints_search_uri'].split("&unique=prints")[0]+"&unique=art"
            self.get_ref_imgs(search_uri)
            
            if(len(self.ref_imgs) == 1):
                self.calib_img_idx = 0
            else:
                self.show_gallery()
    
    def get_ref_imgs(self, search_uri):        
        query_response = requests.get(search_uri).json()
        self.ref_imgs = list(map(lambda x: im_from_url(x['image_uris']['small']), query_response['data']))
        self.resized_imgs = list(map(lambda x: cv2.resize(x, (self.gallery_card_width, self.gallery_card_height), cv2.INTER_AREA), self.ref_imgs))
        self.gallery_page = 0
        self.gallery_page_max = len(self.ref_imgs)//6
    
    def show_gallery(self):
        self.gallery_img = np.zeros((2*self.gallery_card_height, 4*self.gallery_card_width, 3), dtype=np.uint8)   

        for i,resized_img in enumerate( self.resized_imgs[ self.gallery_page*6 : (self.gallery_page+1)*6 ]):
            col = i%3
            row = (i//3)%2
            self.gallery_img[
                row*self.gallery_card_height:(row+1)*self.gallery_card_height, 
                col*self.gallery_card_width:(col+1)*self.gallery_card_width,:] = resized_img
        
        if(self.gallery_page > 0):
            add_text(self.gallery_img, "Previous Page", (int(self.gallery_card_width*3.5), int(self.gallery_card_height*0.5)))
            
        if(self.gallery_page < self.gallery_page_max):
            add_text(self.gallery_img, "Next Page"    , (int(self.gallery_card_width*3.5), int(self.gallery_card_height*1.5)))
            
        cv2.imshow("Gallery", self.gallery_img)
        cv2.setMouseCallback("Gallery", self.make_selection)
        cv2.waitKey(0)
        cv2.destroyWindow("Gallery")
    
    def make_selection(self, event, x, y, flags, param):
    
        if(event == cv2.EVENT_LBUTTONDOWN):
            cv2.setMouseCallback("Gallery", lambda *args: None)
            col = x//self.gallery_card_width
            row = y//self.gallery_card_height
            if(col == 3):
                if  (row==0 and self.gallery_page > 0):
                    self.gallery_page -= 1
                    self.show_gallery()
                elif(row==1 and self.gallery_page < self.gallery_page_max):
                    self.gallery_page += 1
                    self.show_gallery()
                else:
                    cv2.setMouseCallback("Gallery", self.make_selection)
            else:
                idx = self.gallery_page*6+(row*3+col)
                if(idx < len(self.ref_imgs)):
                    self.calib_img_idx = idx
                    cv2.destroyWindow("Gallery")
                else:
                    cv2.setMouseCallback("Gallery", self.make_selection)
    
    
    def select_point(self, event, x, y, flags, param):
       
        if(event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN):    
            frame_copy = param.copy()
            if(event == cv2.EVENT_LBUTTONDOWN):
                if(self.cand_n_points == 4):
                    self.show_cropped = True
                else:
                    self.cand_points[self.cand_n_points] = (x,y)
                    self.cand_n_points += 1
            else:
                if self.cand_n_points!=0:
                    self.cand_n_points -= 1
                    
            for i in range(self.cand_n_points):
                cv2.circle(frame_copy, (self.cand_points[i][0], self.cand_points[i][1]), 2, color=(255,255,255))
            cv2.imshow("Preview", frame_copy)
                
    
    def get_candidate(self):
        if self.search_auto:
            self.get_candidate_auto()
        else:
            self.get_candidate_manual()
    
    def select_candidate_option(self, event, x, y, flags, param):
        if(event == cv2.EVENT_LBUTTONDOWN):
            (candidates, idx, rotate_180) = param
            if(x < self.gallery_card_width):
                if(rotate_180):
                    self.cand_img = cv2.rotate(candidates[idx][0], cv2.ROTATE_180)
                else:
                    self.cand_img, _ = candidates[idx]
                cv2.destroyWindow("Candidates")
            else:
                bounds = np.linspace(0, self.gallery_card_height, num=5, dtype=np.int32)
                
                if(bounds[0]<=y<=bounds[1]):
                    if(idx > 0):
                        self.display_candidate(candidates, idx-1, False)
                elif(bounds[1]<y<=bounds[2]):
                    if(idx != len(candidates)-1):
                        self.display_candidate(candidates, idx+1, False)
                elif(bounds[2]<y<=bounds[3]):
                    self.display_candidate(candidates, idx, not rotate_180)
                else:
                    self.get_candidate_auto()
    
    def display_candidate(self, candidates, idx, rotate_180):
        cand_img, _  = candidates[idx]
        cand_resize = cv2.resize(cand_img, (self.gallery_card_width, self.gallery_card_height), cv2.INTER_AREA)
        if(rotate_180):
            cand_resize = cv2.rotate(cand_resize, cv2.ROTATE_180)
        display_img = np.zeros((self.gallery_card_height, 2*self.gallery_card_width, 3), dtype=np.uint8)
        display_img[:, :self.gallery_card_width, :] = cand_resize
        if(idx > 0):
            add_text(display_img, "Previous Page", (int(self.gallery_card_width*1.5), int(self.gallery_card_height*0.125)))
        if(idx != len(candidates)-1):
            add_text(display_img, "Next Page", (int(self.gallery_card_width*1.5), int(self.gallery_card_height*0.375)))
        add_text(display_img, "Rotate 180", (int(self.gallery_card_width*1.5), int(self.gallery_card_height*0.625)))
        add_text(display_img, "Re-capture", (int(self.gallery_card_width*1.5), int(self.gallery_card_height*0.875)))
        cv2.imshow("Candidates", display_img)
        cv2.setMouseCallback("Candidates", self.select_candidate_option, param=(candidates,idx,rotate_180))
        key = cv2.waitKey(0)
        if(key == 27):
            cv2.destroyWindow("Candidates")
        
    
    
    def get_candidate_auto(self):
        identifier = CardIdentifier((self.cache_name_to_idx, self.cache_names, self.hashes))
        vc = cv2.VideoCapture(1)
        candidates = []
        if vc.isOpened():
            _, frame = vc.read()
            candidates = identifier.get_card_candidates(frame)
        vc.release()
        if(len(candidates) > 0):
            self.display_candidate(candidates, 0, False)
        else:
            print("No candidate cards found")
        
    
    def confirm_crop(self, event, x, y, flags, param):
        if(event == cv2.EVENT_LBUTTONDOWN):
            self.cand_img = param
            cv2.destroyWindow("Crop")
            
        elif(event == cv2.EVENT_RBUTTONUP):
            self.show_cropped = False
            cv2.destroyWindow("Crop")
            self.get_candidate_manual()
            
    
    def get_candidate_manual(self):
        vc = cv2.VideoCapture(1)
        if vc.isOpened():
            rval, frame = vc.read()
            frame = cv2.resize(frame, (np.uint32(self.calib_height/frame.shape[0] * frame.shape[1]), self.calib_height), cv2.INTER_AREA)
            frame_copy = frame.copy()
            for i in range(self.cand_n_points):
                cv2.circle(frame_copy, (self.cand_points[i][0], self.cand_points[i][1]), 2, color=(255,255,255))
            cv2.imshow("Preview", frame_copy)
            cv2.setMouseCallback("Preview", self.select_point, param=frame)
            while(True):
                if(self.show_cropped or cv2.waitKey(100) == 27):
                    break
        vc.release()
        cv2.destroyWindow("Preview")
        if(self.show_cropped):
            warped = four_point_transform(frame, self.cand_points)
            cropped_img = np.zeros((self.gallery_card_height, 2*self.gallery_card_width, 3), dtype=np.uint8)
            cropped_img[:,                       :self.gallery_card_width,:] = self.resized_imgs[self.calib_img_idx]
            cropped_img[:,self.gallery_card_width:                       ,:] = cv2.resize(warped, (self.gallery_card_width, self.gallery_card_height), cv2.INTER_AREA)
            cv2.imshow("Crop", cropped_img)
            cv2.setMouseCallback("Crop", self.confirm_crop, param=warped)
            cv2.waitKey(0)
        
    
    
    
    def calibrate_offsets(self):
        if(self.cand_img is None):
            return None
        removed_hashes = np.delete(self.hashes, self.cache_name_to_idx[self.calib_name], axis=0)
        
        other_hashes = np.split(removed_hashes, 3, axis=1)
        
        ref_img = self.ref_imgs[self.calib_img_idx]
        
        predict_offsets = np.int8(np.array(cv2.mean(ref_img)) - np.array(cv2.mean(self.cand_img)))[:-1]
        best_offsets = np.int8([255,255,255])
        best_diffs = np.zeros((3))
        
        for ch in range(3):
            ref_channel = self.ref_imgs[self.calib_img_idx][:,:,ch]
            ref_phash = phash_dim(ref_channel)
            cand_channel = self.cand_img[:,:,ch]
                
            for offset in np.arange(predict_offsets[ch]-1, predict_offsets[ch]):
                cand_offset = apply_offset(cand_channel, offset)
                #cand_offset = cv2.add(cand_channel, np.full(cand_channel.shape, offset, dtype=np.int8), dtype=cv2.CV_8U)
                cand_phash = phash_dim(cand_offset)
                
                dist = np.count_nonzero(cand_phash != ref_phash)
                
                
                other_dists = np.count_nonzero(cand_phash != other_hashes[ch], axis=1)
                avg = np.mean(other_dists)
                std = np.std(other_dists)
                diff_measure = (avg - dist)/std
                if(diff_measure > best_diffs[ch] or
                  (diff_measure == best_diffs[ch] and np.abs(offset) < np.abs(best_offsets[ch])) ):
                    best_offsets[ch] = offset
                    best_diffs[ch] = diff_measure
        return best_offsets
    def calibrate_trim(self):
        if(self.cand_img is None):
            print("No candidate image found")
            return None
        removed_hashes = np.delete(self.hashes, self.cache_name_to_idx[self.calib_name], axis=0)
        ref_img = self.ref_imgs[self.calib_img_idx]
        ref_phash = phash(ref_img)
        
        best_diff = 0
        best_trim = 1.
        for new_trim in np.linspace(0.8,1.,20):
        
            cand_trimmed = apply_trim(self.cand_img, new_trim)
            cand_phash = phash(cand_trimmed)
            
            dist = np.count_nonzero(cand_phash != ref_phash)
            other_dists = np.count_nonzero(cand_phash != removed_hashes, axis = 1)
            avg = np.mean(other_dists)
            std = np.std(other_dists)
            diff_measure = (avg-dist)/std
            if(diff_measure >= best_diff):
                best_trim = new_trim
                best_diff = diff_measure
        return best_trim
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("code", help="code of the cache to use")
    parser.add_argument("calib_card", help="card name to calibrate")
    parser.add_argument("-auto", action="store_true", help="automatically detect card")
    args = parser.parse_args()
    
    cache_code = args.code
    calib_name = args.calib_card
    
    if(not cache_exists(cache_code)):
        print("Cache does not exist", file=sys.stderr)
        exit(1)
        
    calibrator = Calibrator(load_cache(cache_code), calib_name, args.auto)
    print(calibrator.calibrate())
    