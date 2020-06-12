import cv2 as cv
import numpy as np
import math
import random
import os

# 一次偵測到特徵點的數量
MAX_FEATURES = 500
# 配對比率
GOOD_MATCH_PERCENT = int(MAX_FEATURES*0.2)
count = 0

def load_img():
    print("Load Image.....")
    pic_1 = cv.imread('./data/View-1.jpg')
    pic_2 = cv.imread('./data/View-2.jpg')
    pic_3 = cv.imread('./data/View-3.jpg')


    
    H = int(pic_1.shape[0])
    W = int(pic_1.shape[1])
    
    # # 將圖片轉換為同一大小
    # pic_1 = cv.resize(pic_1, (H, W))
    # pic_2 = cv.resize(pic_2, (H, W))
    # pic_3 = cv.resize(pic_3, (H, W))
    
    return pic_1, pic_2, pic_3

# match左右兩圖的特徵點 
def self_match(descriptors1, descriptors2, length):
    print("Self matching.......")
    desNum = descriptors1.shape[1]
    
    matches = np.zeros((length), dtype=np.object)
    fp = open("self_matches.txt", "w")

    # 藉由 Euclidean Distance 來得出兩張圖對應的特徵點
    for i in range(length):
        match = cv.DMatch()
        total_min = 9999
        for j in range(length):
            total = 0
            for k in range(desNum):
                total = total + np.square(descriptors1[j][k]-descriptors2[i][k])
            total = math.sqrt(total)
            if total_min > total:
                total_min = total
                queryIdx = j
                trainIdx = i
        
        # 將被對到的資訊加入物件中
        match.queryIdx = queryIdx
        match.trainIdx = trainIdx
        match.distance = total_min

        matches[i] = match
    

    matches = matches.tolist()
    matches.sort(key=lambda x: x.distance, reverse=False)
    # 取得前20%的配對特徵點作為優良特徵點
    good_matches = matches[:GOOD_MATCH_PERCENT]
    for match in good_matches:
        fp.write("({}, {}) Distance: {}\n".format(str(match.queryIdx), str(match.trainIdx),  str(match.distance)))
    fp.close()
    return good_matches
    
# 繪製配對的特徵點
def drawMatches(img1, kp1, img2, kp2, matches):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    drawPoints = np.zeros((max([rows1,rows2]),cols1 + cols2, 3),dtype = 'uint8')
    # 拼接圖像
    drawPoints[:rows1, :cols1] = np.dstack([img1])
    drawPoints[:rows2, cols1:] = np.dstack([img2])
    
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
        
        # 圈出配對到的特徵點 
        cv.circle(drawPoints, (int(x1),int(y1)),4,(255,255,0),1)
        cv.circle(drawPoints,(int(x2)+cols1,int(y2)),4,(0,255,255),1)
        
        # 將配對的特徵點連線
        cv.line(drawPoints,(int(x1),int(y1)),(int(x2)+cols1,int(y2)),(255,0,0),1)
    cv.imshow("test", drawPoints)
    cv.imwrite("drawMatches{}.jpg".format(count), drawPoints)
    cv.waitKey(0)

# 找出Homography martix
def findHomography(queryPt, trainPt):
    A = np.zeros((8,9), dtype=np.float32)
    H = np.zeros((3,3), dtype=np.float32)
    count = 0

    # 套用8 dof 公式
    for i in range(4):
        A[count] = [ trainPt[i][0], trainPt[i][1], 1, 0, 0, 0, -trainPt[i][0]*queryPt[i][0], -trainPt[i][1]*queryPt[i][0], -queryPt[i][0] ]
        A[count+1] = [ 0, 0, 0, trainPt[i][0], trainPt[i][1], 1, -trainPt[i][0]*queryPt[i][1], -trainPt[i][1]*queryPt[i][1], -queryPt[i][1] ]
        count += 2
  
    u, s, vh = np.linalg.svd(A, full_matrices=True)  
    L = vh[-1,:] / vh[-1,-1]
    H = L.reshape(3, 3)
    
    return H

# 利用RANSC找出最好的 Homography martix
def best_H(matches, keypoints1, keypoints2):
    print("Find Best H.....")
    queryPt = np.zeros((len(matches),2), dtype=np.float32)
    trainPt = np.zeros((len(matches),2), dtype=np.float32)
    fp = open("Homography.txt", "w")
    # 列出配對好的特徵點座標
    for i in range(len(matches)):
        queryPt[i] = keypoints1[matches[i].queryIdx].pt
        trainPt[i] = keypoints2[matches[i].trainIdx].pt
        fp.write("queryPt: {} trainPt: {}\r\n".format(str(queryPt[i]), str(trainPt[i])))
    fp.close()
   
    sampleNum = len(matches) # match features 總數
    runCount = 10000 # RANSAC 次數
    threshold = 5.0 
    throwInNum = 4 # 丟入feature數量
    InlierMax = 0 # 紀錄 Inlier Points 最大值
    bestH = None # 紀錄最好的 Homography martix
    
    run = np.zeros(runCount)
    for i in range(runCount):
        
        # 隨機選出四個 feature points
        randomIdx = random.sample(range(sampleNum), throwInNum)
        # 將隨機選出的四個特徵點丟入 findHomography() ，得出一個 Homography Martix
        H = findHomography(queryPt[randomIdx], trainPt[randomIdx])

        # 紀錄這次 Inlier Points 數量
        InlierNum = 0
        for j in range(sampleNum):
            coor= np.array([ trainPt[j][0], trainPt[j][1], 1 ])
            wraped = H @ coor
            wraped = wraped / wraped[2]

            # 利用 Euclidean Distance 來計算轉換過的結果
            distance = np.linalg.norm(wraped[:2] - queryPt[j])
            
            if(distance < threshold):
                InlierNum = InlierNum+1
        if(InlierMax < InlierNum):
            InlierMax = InlierNum
            bestH = H
            print(InlierMax)
        run[i] = InlierMax

    print(run)
   
    return bestH

# 利用Homography martix進行圖片的轉換
def self_warp(trainImg, h, size):
    height,  width = size
    result = np.zeros((height, width, 3), dtype=np.uint8)

    img_h, img_w = trainImg.shape[:2]
    h_inv = np.linalg.inv(h)

    for i in range(height):
        for j in range(width):
            coor = np.array([[j], [i], [1]])
            train_coor = h_inv.dot(coor)
            x = int(train_coor[0]/train_coor[2])
            y = int(train_coor[1]/train_coor[2])

            if(x >= 0 and y >= 0) and (x <img_w and y <img_h):
                result[i][j] = trainImg[y][x]
    
    return result

def stitchImage(img_1, img_2):
    # 轉為灰階
    gray1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

    # 取得兩張圖片的特徵點
    detector = cv.ORB_create()
    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)
    keypointsNum = len(keypoints1)

    # 進行feature的比對並找出事配對良好的特徵點
    good_matches = self_match(descriptors1, descriptors2, keypointsNum)
   
    # 畫出配對好的特徵點
    drawMatches(img_1, keypoints1, img_2, keypoints2, good_matches)

    # 找到最好的Homogphy martix
    BH = best_H(good_matches, keypoints1, keypoints2)

    # 去除黑邊
    cut = True
    width = 0
    for i in range(img_1.shape[1]):
        g, r, b = img_1[0][i]
        if (g ==0 and r==0 and b==0 and cut):
            cut = False
            width = i
        elif(g !=0 or r!=0 or b!=0):
            cut = True
            width = 0
    if(cut and width==0):
        print("Not Cut")
        width = img_1.shape[1]
    print(img_1.shape)
    size = (img_1.shape[0], width+img_2.shape[1])

    wrap_image = self_warp(img_2, BH, size)
    cv.imshow('warp', wrap_image)
    stitchedImage = np.copy(wrap_image)
    cv.waitKey(0)

    for i in range(img_1.shape[0]):
        for j in range(width):
            stitchedImage[i][j] = img_1[i][j]
    # stitchedImage[0:img_1.shape[0],0:width] = img_1[0:img_1.shape[0]][0:width]
    
    return stitchedImage


def main():
    # 取得輸入影像
    img_1, img_2, img_3= load_img()
    count = 0
    # 進行圖1及圖2的影像拼接
    stitchedImage = stitchImage(img_1, img_2)
    cv.imshow("stitchedImage",stitchedImage)
    cv.imwrite("stitchedImage.jpg", stitchedImage)
    cv.waitKey(0)
    count = 1
    stitchedImage2 = stitchImage(stitchedImage, img_3)
    cv.imshow("stitchedImage2",stitchedImage2)
    cv.imwrite("stitchedImage2.jpg", stitchedImage2)
    cv.waitKey(0)



main()