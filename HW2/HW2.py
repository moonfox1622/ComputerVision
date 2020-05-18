import cv2 as cv
import numpy as np
import json 

cv.ocl.setUseOpenCL(False)
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.1
def load_img():
    pic_1 = cv.imread('./data/Wilde-1.jpg')
    pic_2 = cv.imread('./data/Wilde-2.jpg')
    pic_3 = cv.imread('./data/Wilde-3.jpg')

    
    H = int(pic_1.shape[0]*0.2)
    W = int(pic_1.shape[1]*0.2)
    

    pic_1 = cv.resize(pic_1, (H, W))
    pic_2 = cv.resize(pic_2, (H, W))
    pic_3 = cv.resize(pic_3, (H, W))
    
    return pic_1, pic_2, pic_3

def feature(gray1, gray2):
    detector = cv.ORB_create()

    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)
    # keypoints3, descriptors3 = detector.detectAndCompute(gray3, None)

    imgKeypoints1 = np.array([])
    imgKeypoints2 = np.array([])
    imgKeypoints3 = np.array([])

    imgKeypoints1 = cv.drawKeypoints(gray1, keypoints1, imgKeypoints1, color=(0,0,255),flags=0)
    imgKeypoints2 = cv.drawKeypoints(gray2, keypoints2, imgKeypoints2, color=(0,0,255),flags=0)
    # imgKeypoints3 = cv.drawKeypoints(gray3, keypoints3, imgKeypoints3, color=(0,0,255),flags=0)
    
    test = json.dumps(keypoints1[0])
    print(test)
    # print(descriptors1[0])
    cv.imwrite("keypoint1.jpg", imgKeypoints1)
    cv.imwrite("keypoint2.jpg", imgKeypoints2)
    # cv.imwrite("keypoint3.jpg", imgKeypoints3)
    
    return keypoints1, descriptors1, keypoints2, descriptors2

def match(img_1, img_2, keypoints1, descriptors1, keypoints2, descriptors2):
    # 特徵匹配.
  matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  # print(descriptorsL, descriptorsR)
  matches = matcher.match(descriptors1, descriptors2, None)
#   print(keypoints1.shape)
  print (descriptors1.shape)
  # 每個匹配到的特徵帶有分數，由小至大排序
  matches.sort(key=lambda x: x.distance, reverse=False)

  # 僅保留前15%分數較高的匹配特徵
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # 繪出兩張相片匹配的特徵點
  imMatches = cv.drawMatches(img_1, keypoints1, img_2, keypoints2, matches, None)
  print("Saving Image with matches"); 
  cv.imwrite("matches.jpg", imMatches)

# def self_match():


def main():
    
    img_1, img_2, img_3 = load_img()

    gray1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
    gray3 = cv.cvtColor(img_3, cv.COLOR_BGR2GRAY)

    # keypoints1, descriptors1, keypoints2, descriptors2 = feature(gray1, gray2)
    detector = cv.ORB_create()

    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)
    keypoints3, descriptors3 = detector.detectAndCompute(gray3, None)

    match(img_1, img_2, keypoints1, descriptors1, keypoints2, descriptors2)

    cv.imshow('pic1', img_1)
    cv.imshow('pic2', img_2)
    cv.imshow('pic3', img_3)
    cv.waitKey(0)

main()