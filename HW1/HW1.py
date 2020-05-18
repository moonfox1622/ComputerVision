import cv2 as cv
import numpy as np
import math
 
def process(floder):
    light_pos = np.zeros((6,3),dtype=float)
    
    light = open("./data/{}/light.txt".format(floder))
    light = light.readlines()

    for i in range(0,6):

        picture_name = "./data/{}/pic{}.bmp".format(floder,i+1) 
        picture_gray = cv.imread(picture_name,0)
        # 轉為灰階
        # picture_gray = cv.cvtColor(picture, cv.COLOR_BGR2GRAY)
        picture_list = np.array(picture_gray, dtype=np.uint8)
        H,W = picture_gray.shape
        
        if i==0 :
            #預計存進6張圖片的灰階值
            rgb = np.zeros((6,H,W),dtype=np.uint8) 
        
        #將進行過灰階處過的圖片(picture_gray)的灰階值存進rgb中
        for j in range (picture_list.shape[0]): 
            for k in range (picture_list.shape[1]): 
                # print (picture_list[j][k])
                rgb[i][j][k] = picture_gray[j][k]
        np.savetxt("{}.txt".format(i),rgb[i], fmt='%3d ')
        
        # 處理light.txt
        dot1 = light[i].find(',')
        dot2 = light[i].find(',',dot1+1)
        x = int(light[i][light[i].find('(')+1:dot1])
        y = int(light[i][dot1+1:dot2])
        z = int(light[i][dot2+1:light[i].find(')')])
        
        # 計算光的向量
        lx = x/math.sqrt(np.square(x) + np.square(y) + np.square(z))
        ly = y/math.sqrt(np.square(x) + np.square(y) + np.square(z))
        lz = z/math.sqrt(np.square(x) + np.square(y) + np.square(z))
        light_pos[i] =[lx, ly, lz]
        
    # print(light_pos)
    return rgb, light_pos

def Lambertian_Surfaces(picture, light_pos):
    # 將光的向量轉換成逆矩陣
    light_T = np.linalg.pinv(light_pos)
    #print(light_T)
    # light_T = np.transpose(light_pos)
    number, H, W = picture.shape
    orientation = np.zeros((H, W, 3), dtype=np.uint8)
    albedo = np.zeros((H, W), dtype=np.uint8)
    Big_N = np.zeros((3),dtype=np.int8)
    # 套用公式將反矩陣後的光的向量(light_T)乘上六張圖片的同一pixel，迴圈的大小為一張圖片的大小
    for i in range(H):
        for j in range(W):
            I = np.zeros((6))
            for k in range(number):
                I[k] = picture[k][i][j]
                
            Big_N = light_T @ I
            orientation[i][j] = Big_N
            # 算出orientation後將結果套公式求出albedo
            albedo[i][j] = math.sqrt( np.square(Big_N[0]) + np.square(Big_N[1]) + np.square(Big_N[2]))
            if albedo[i][j] == 0:
                orientation[i][j] = [0,0,0]
            else:
                for k in range(3):
                    N = Big_N[k]
                    orientation[i][j][k] = ((N/albedo[i][j])+1)*100
            # if Big_N[0]!=0:
            #     print(Big_N)
            #     print(albedo[i][j])
            #     print(orientation[i][j])
            #     #print(light_T)
            #     #print(Big_N)
            #     exit()
                
        
        
    return orientation, albedo
    
def main():
    floder = 'teapot'
    picture, light_pos = process(floder)
    
    orientation, albedo = Lambertian_Surfaces(picture, light_pos)

    # 將orientation, albedo以圖片的方式進行輸出
    cv.imshow("albedo",albedo)
    cv.imshow("orientation",orientation)
    cv.imwrite("albedo.jpg", albedo)
    cv.imwrite("orientation.jpg",orientation)
    cv.waitKey(0)
    np.savetxt("albedo.txt",albedo, fmt='%d ')


main()