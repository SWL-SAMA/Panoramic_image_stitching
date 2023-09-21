import cv2
import os
import numpy as np
import math
from tqdm import tqdm

# 寻找SIFT特征的函数，返回值kp为关键点的集合，des是提取出的n*128维特征向量
def sift_kp(image):
    out_image = image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


# 输入两张图像，返回Flann匹配SIFT特征点连线图
def Match_and_Draw(img_1, img_2):
    kp1, des1 = sift_kp(img_1)
    kp2, des2 = sift_kp(img_2)
    # 获取flann匹配器
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # searchParams 指定递归遍历的次数，值越高结果越准确，但是消耗的时间也越多。
    searchParams = dict(checks=50)
    # 使用FlannBasedMatcher 寻找最近邻近似匹配
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    # 使用knnMatch匹配处理，并返回匹配matches
    matches = flann.knnMatch(des1, des2, k=2)
    # 通过掩码方式计算有用的点
    matchesMask = [[0, 0] for i in range(len(matches))]  # 定义一个长度与match匹配的空配对
    # 通过描述符的距离进行选择需要的点
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:  # 通过0.5系数来决定匹配的有效关键点数量
            matchesMask[i] = [1, 0]
    drawPrams = dict(matchColor=(255, 0, 0), singlePointColor=(0, 0, 255), matchesMask=matchesMask, flags=0)
    # 匹配结果图片
    good_match = []  # 使用一个空数组存储有效关键点下标
    for m in matches:
        if len(m) == 2 and m[0].distance < 0.5 * m[1].distance:
            good_match.append((m[0].queryIdx, m[0].trainIdx))
    # 将找到的有效匹配数组转换为float型数据类型，转换为x，y坐标形式
    kp1_float = np.float32([kp.pt for kp in kp1])
    kp2_float = np.float32([kp.pt for kp in kp2])
    # 转换成x，y坐标值形式
    kp1_float = np.float32([kp1_float[a[0]] for a in good_match])
    kp2_float = np.float32([kp2_float[a[1]] for a in good_match])
    res = cv2.drawMatchesKnn(img_1, kp1, img_2, kp2, matches, None, **drawPrams)  # 绘制匹配结果
    # cv2.imshow('good_matches', res)
    return res, good_match, kp1_float, kp2_float


def cylindrical_projection(img, f):  # 将图像进行柱面变换，有效解决多张图像拼接时产生的分辨率下降问题,f代表曲率，越小曲率越大
    rows = img.shape[0]
    cols = img.shape[1]

    blank = np.zeros_like(img)
    center_x = int(cols / 2)
    center_y = int(rows / 2)

    for y in range(rows):
        for x in range(cols):
            theta = math.atan((x - center_x) / f)
            point_x = int(f * math.tan((x - center_x) / f) + center_x)
            point_y = int((y - center_y) / math.cos(theta) + center_y)

            if point_x >= cols or point_x < 0 or point_y >= rows or point_y < 0:
                pass
            else:
                blank[y, x, :] = img[point_y, point_x, :]
    return blank

def calWeight(d, k):  # d为重合部分的直径，k是融合计算权重参数，返回权重值y
    x = np.arange(-d / 2, d / 2)
    y = 1 / (1 + np.exp(-k * x))  # Sigmoid函数作为权值分布的计算式
    return y

def Stitching(img_1, img_2):  # img_1为左图，img_2为右图
    row1, col1, chl1 = img_1.shape
    row2, col2, chl2 = img_2.shape
    res, g_m, kp1s, kp2s = Match_and_Draw(img_1, img_2)
    # 求解转换矩阵
    M, status = cv2.findHomography(kp2s, kp1s, cv2.RANSAC, 4.0)
    img_2 = cv2.warpPerspective(img_2, M, (
        img_1.shape[1] + img_2.shape[1], max(img_1.shape[0], img_2.shape[0])))  # 将img_2进行透视变换，依据是上一步求出的转换矩阵
    # Result[0:img_1.shape[0], 0:img_1.shape[1]] = img_1  # 这里原本是直接将变换后的img2拼接到img1右侧，无法解决接缝问题
    # 更改后如下
    overlap = int(img_1.shape[1] / 2)  # 连接两张图片中间的权值过度部分的重合宽度
    # 经过多次测试，overlap最优值的选取与图像的尺寸有关，如果尽量取得很大会使得按权值连接出的“重影”现象有所缓解，但是过大会导致连接处的图像信息损失较为严重，表现为图像变得模糊
    # 经过多次调试，选取最合适的重叠部分宽度为左图列数的1/2

    w = calWeight(overlap, 0.05)  # 经多次参数调整，k = 0.05最合适
    img1_bak = img_1
    img2_bak = img_2
    # 创建一个新的矩阵储存拼接后的图像
    img_new_dst = np.zeros((max(row1, row2), col1 + col2, chl1))
    for i in range(chl1):  # 对三个通道的图像分别进行操作,i代表通道数
        img_1 = img1_bak[:, :, i]
        img_2 = img2_bak[:, :, i]
        img_new = np.zeros((max(row1, row2), col1 + col2))  # 对第i个通道创建一个矩阵，储存该通道的拼接结果
        img_new[0:row1, 0:col1] = img_1  # 先将左图赋值到新矩阵的左侧
        w_expand = np.tile(w, (max(row1, row2), 1))  # 权重扩增
        img_new[0:max(row1, row2), (col1 - overlap):col1] = (1 - w_expand) * img_1[0:row1,
                                                                             (col1 - overlap):col1] + w_expand * img_2[
                                                                                                                 0:row1,
                                                                                                                 (
                                                                                                                         col1 - overlap):col1]
        img_new[:, col1:] = img_2[:, col1:]
        img_new_dst[:, :, i] = img_new
    img_new_dst = np.uint8(img_new_dst)
    return img_new_dst

# 去除图像的黑边并实现裁切,这样能够解决多个图像拼接时出现的问题
def crop_black(img):
    b = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)  # 调整裁剪效果
    binary_image = b[1]  # 二值图--具有三通道
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    x = binary_image.shape[0]
    y = binary_image.shape[1]
    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(y):
            if binary_image[i][j] == 255:
                edges_x.append(i)
                edges_y.append(j)

    left = min(edges_x)  # 左边界
    right = max(edges_x)  # 右边界
    width = right - left  # 宽度
    bottom = min(edges_y)  # 底部
    top = max(edges_y)  # 顶部
    height = top - bottom  # 高度
    pre1_picture = img[left:left + width, bottom:bottom + height]  # 图片截取
    row, col, chl = pre1_picture.shape  # 暂存图像的长宽信息和通道数
    res_img = np.zeros((row + 1, col, chl))
    res_img[0:row, 0:col, 1] = pre1_picture[0:row, 0:col, 1]
    res_img[row, 0:col, 1] = pre1_picture[row - 1, 0:col, 1]

    res_img[0:row, 0:col, 2] = pre1_picture[0:row, 0:col, 2]
    res_img[row, 0:col, 2] = pre1_picture[row - 1, 0:col, 2]

    res_img[0:row, 0:col, 0] = pre1_picture[0:row, 0:col, 0]
    res_img[row, 0:col, 0] = pre1_picture[row - 1, 0:col, 0]

    res_img = np.uint8(res_img)  # 转变数据类型
    return res_img  # 返回图片数据

def load_imgs(imgs_dir):
    imgs_name_list=os.listdir(imgs_dir)
    imgs_list=[]
    for img in imgs_name_list:
        path=imgs_dir+img
        Image=cv2.imread(path)
        imgs_list.append(Image)
    print('Loaded Images')
    return imgs_list

def resize_imgs(img_list):
    resized_list=[]
    for img in img_list:
        img_resized=cv2.resize(img, (500, 500))
        resized_list.append(img_resized)
    print('Images Resized')
    return resized_list

def cylindrical_imgs(img_list):
    cylindrical_list=[]
    for img in img_list:
        c_img=cylindrical_projection(img, 500)
        cylindrical_list.append(c_img)
    print('Cylindrical Images')
    return cylindrical_list

if __name__=="__main__":
    #  读入局部图像
    img_list=load_imgs('./imgs/')

    #  预处理图像，调整大小为500*500
    img_list=resize_imgs(img_list)

    #  柱面变化
    img_list=cylindrical_imgs(img_list)

    #  拼接全景图
    result=None
    for i in tqdm(range(len(img_list))):
        if i !=4:
            if result is None:
                result = Stitching(img_list[i], img_list[i+1])
            else:
                result=Stitching(result, img_list[i+1])
            result = crop_black(result)
        i+=1
    print('Progress Finish')
    cv2.namedWindow('result', 0)
    cv2.resizeWindow('result', 1800, 500)
    cv2.imshow('result', result)
    save_dir='./result_final.jpg'
    cv2.imwrite(save_dir, result)
    print('Saved result to ', save_dir)
    cv2.waitKey(0)
