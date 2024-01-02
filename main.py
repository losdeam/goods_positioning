import cv2
import os 
import numpy as np
import shutil
import time 
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim

img_part_size = 50
num_clusters = 5
# 使用 BRISK 特征检测器
brisk = cv2.SIFT_create()
bf = cv2.BFMatcher()
def count_time(f):
    def warrp(*a,**b):
        t1 = time.time()
        t = f(*a,**b)
        print(f.__name__,"耗时",time.time()-t1)
        return t
    return warrp

def get_cluster_centers(img,show = False ):
    '''
    获取图像中关键点的聚类中心
    input:
        img:图像
        num_clusters : 返回的聚类中心的个数
        show : 是否展示获取到的关键点
    output:
        cluster_centers: 聚类中心的坐标
    '''
    # 在图像上检测关键点
    keypoints, _ = brisk.detectAndCompute(img, None)
    keypoint_coordinates = np.array([kp.pt for kp in keypoints])
    kmeans = KMeans(n_clusters=num_clusters,n_init=10 ,random_state=42)
    kmeans.fit(keypoint_coordinates)
    # 获取聚类中心的坐标,
    cluster_centers = kmeans.cluster_centers_
    if show : 
        # 在原图上标记检测点
        img_with_keypoints = img.copy()
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, img_with_keypoints)

        # 在原图上标记聚类中心
        for center in cluster_centers:
            center_coordinates = tuple(map(int, center))
            img_with_keypoints = cv2.circle(img_with_keypoints, center_coordinates, 10, (0, 255, 0), -1)

        # 显示带有标记的图像
        cv2.imshow('Image with Keypoints and Cluster Centers', img_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cluster_centers

def get_center_img(orign_resize_img,center_list,show = False):
    '''
    根据所给坐标在图像中截取一定大小的片段
    input:
        orign_img:原始图像
        center_list : 该图聚类中心的列表
        img_part_size:截取图像的大小(由聚类中心向四周延长的距离)
    output:
        cluster_centers: 聚类中心的坐标
    '''
    cluster_centers_img_list = []
    border_list = []
    for center_x,center_y in center_list:
        shape_x,shape_y,_ = orign_resize_img.shape
        # 计算图像边界坐标
        x_min = int(center_x - img_part_size)  if center_x - img_part_size>= 0 else 0 
        x_max = int(center_x + img_part_size)  if center_x + img_part_size< shape_x else shape_x
        y_min = int(center_y - img_part_size)  if center_y - img_part_size>= 0 else 0 
        y_max = int(center_y + img_part_size)  if center_y + img_part_size< shape_y else shape_y 
        # 计算填充值
        extend_x = 2 * img_part_size - x_max + x_min
        extend_y = 2 * img_part_size - y_max + y_min
        if extend_x:
            if x_min :
                x_min -= extend_x
            else:
                x_max += extend_x

        if extend_y:
            if y_min :
                y_min -= extend_y
            else:
                y_max += extend_y
        # 根据图像边界坐标在原图中进行截图
        cluster_centers_img = orign_resize_img[y_min:y_max,x_min:x_max]
        if show:
            img_with_bbox = orign_resize_img.copy()
            cv2.rectangle(img_with_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # (0, 255, 0) 是矩形框的颜色，2 是线宽
            cv2.imshow("cluster_centers_img",cluster_centers_img)
            cv2.imshow("gray_img",orign_resize_img)
            cv2.imshow("img_with_bbox",img_with_bbox)
            cv2.waitKey(0)
        cluster_centers_img_list.append(cluster_centers_img)
        border_list.append([x_min,x_max,y_min,y_max])
    return cluster_centers_img_list,border_list
@count_time
def find_goods_centers(bg,cluster_center_img_list):
    '''
    寻找每张图的商品所在的聚类中心
    input:
        bg : 灰色背景图
        cluster_center_img_list：聚类中心的图片列表
    '''
    # 先通过一次遍历来寻找第一张图中商品的聚类中心
    #--------------------------
    #直接通过判断聚类中心与背景的相似度即可
    # 越不像的越可能是商品位置
    #--------------------------
    n = len(cluster_center_img_list)
    # 寻找相似度最低的聚类中心
    match_list = [1]*n
    good_cluster_list = [None] *n
    # 遍历图像
    for i in range(n):
        min_match = 1
        #遍历当前图像的所有聚类中心图像
        for index,cluster_center_img in enumerate(cluster_center_img_list[i]):
                # 提取当前图像的关键点和描述符
                cluster_center_img_resize = cv2.resize(cluster_center_img, (320,320))
                similarity = ssim(bg, cluster_center_img_resize,channel_axis =2 )
                if  similarity <= min_match:
                    min_match = similarity
                    match_list[i] = index
        good_cluster_list[i] = cluster_center_img_list[i][match_list[i]]
    return match_list,good_cluster_list
#-------------------------二分算法v1.0 -------------------------
@count_time
def get_goods(bg_reisze_img,good_index_list,orign_resize_img_list,cluster_centers_list,cluster_border_list  ,size=(320,320),is_show= False  ):
    '''
    根据获取到的商品聚类中心来寻找商品整体
    input :
        bg_reisze_img : 经过大小调整的背景图
        good_index_list : 商品的聚类中心下标
        orign_resize_img_list : 经过大小调整的原图列表
        cluster_centers_list : 图像聚类中心列表
        cluster_border_list : 图像聚类中心图像边界值
        size : 原图大小
        is_show : 是否显示二分过程
    output : 
        result : 保存所有商品边界值的列表
    '''
    result = []
    for index , i in enumerate(good_index_list):
        orign_resize_img = orign_resize_img_list[index]
        center_x,center_y = map(int,cluster_centers_list[index][i])
        left_x,right_x,left_y,right_y = cluster_border_list[index][i]
        left_x = binary (bg_reisze_img,orign_resize_img,(left_y,right_y) ,center_x,size=size,is_left =  False , is_x = True,is_show= is_show)
        right_x = binary (bg_reisze_img,orign_resize_img,(left_y,right_y) ,center_x,size=size,is_left = True ,is_x = True, is_show= is_show)
        if is_show: 
            print(1111)
            cv2.imshow('1',orign_resize_img[:,left_x:right_x,:])
            cv2.waitKey(0)
        left_y = binary (bg_reisze_img,orign_resize_img,(left_x,right_x) ,center_y,size=size,is_left = False , is_x = False,is_show= is_show)
        right_y = binary (bg_reisze_img,orign_resize_img,(left_x,right_x) ,center_y,size=size,is_left = True, is_x = False,is_show= is_show)
        if is_show:
            print(index)
            cv2.imshow('test',orign_resize_img[left_y:right_y,left_x:right_x,:])
            cv2.waitKey(0)
        result.append((left_x,right_x,left_y,right_y))
    return result




    return result
def binary (bg_reisze_img,orign_resize_img,side ,site,size=(320,320),is_left = False , is_x = True,is_show =False):
    """
    input :
        bg_reisze_img : 经过大小调整的背景图
        orign_resize_img : 经过大小调整的原图
        side : 当前二分轴的另一轴的最大最小值,默认为聚类中心裁剪图像的大小
        site : 聚类中心的坐标位置
        size : 图像大小
        is_left : 模式标志,当前所给site是否为left,left = site or 0 
        is_x : 模式标志,当前二分轴
        is_show: 是否展示二分过程
    output :
        left : 二分结果
    """
    if is_left :
        left = site 
        right = size[0] if is_x else size[1] 
    else:
        left = 0 
        right = site
    while left < right :
        mid = (left+right )//2 
        if right - mid  <= 7 :
            break
        if is_x :
            img_r =  orign_resize_img[side[0]:side[1],mid:right,:] if is_left else orign_resize_img[side[0]:side[1],left:mid,:]
            img_bg = bg_reisze_img[side[0]:side[1],mid:right,:] if is_left else bg_reisze_img[side[0]:side[1],left:mid,:] 
            if is_show:
                t1 = orign_resize_img.copy()
                t2 = orign_resize_img.copy()
                cv2.rectangle(t1, (mid, side[0]), (right, side[1]), (0, 255, 0), 2)
                cv2.rectangle(t2, (left, side[0]), (mid, side[1]), (0, 255, 0), 2)
        else:
            img_r = orign_resize_img[mid:right,side[0]:side[1],:] if is_left else orign_resize_img[left:mid,side[0]:side[1],:]
            img_bg = bg_reisze_img[mid:right,side[0]:side[1],:] if is_left else bg_reisze_img[left:mid,side[0]:side[1],:]
            if is_show:
                t1 = orign_resize_img.copy()
                t2 = orign_resize_img.copy()
                cv2.rectangle(t1, (side[0],mid), (side[1],right ), (0, 255, 0), 2)
                cv2.rectangle(t2, (side[0],left), (side[1],mid), (0, 255, 0), 2)

        similarity = ssim(img_r, img_bg,channel_axis =2)    
        if is_show:
            print(mid,right,side[0],side[1],is_x,is_left)
            cv2.imshow("l",t1)
            cv2.imshow("r",t2)
            cv2.imshow("img_r",img_r)
            cv2.imshow('img_bg',img_bg)
            cv2.waitKey(0)

            print(img_r.shape,similarity)
        if similarity > 0.90 :
            if is_left:
                right = mid 
            else:
                left = mid 
        else:
            if is_left:
                left = mid 
            else:
                right = mid 
    return left 
#------------------------- -------------------------
@count_time
def resize_list(img_list):
    result= []
    if len(img_list[0]) ==2 :
        for  img ,path in img_list:
            img = cv2.resize(img,(320,320))
            # print(path)
            cv2.imwrite(path,img)
            result.append(img)
    else:
        for  img in img_list:
            # print(img)
            img_ = cv2.imread("image_pre" +"/" + img)
            img_ = cv2.resize(img_,(320,320))
            result.append(img_)
    return result
@count_time
def image_delete_local(folder_path):
    """
    删除本地数据
    """
        # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 获取文件夹下的所有文件和子文件夹
        items = os.listdir(folder_path)

        # 删除文件夹下的所有文件和子文件夹
        for item in items:
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

        print(f"文件夹 '{folder_path}' 已清空。")
    else:
        print(f"文件夹 '{folder_path}' 不存在。")
@count_time
def image_from_video( target_frame_count ,video=None ,output_folder = 'image_pre',video_flie_path='video'):
    '''
    从视频中获取图像数据，
    '''
    if  video:
        video_path = video_flie_path + '/' + '1.mp4'
        with open(video_path, 'wb') as f:
            f.write(video.read())
    video_dir = os.listdir(video_flie_path)
    output_list = []
    # print(video_dir)
    for index , video_name in  enumerate(video_dir):
        if video_name[-4:] != ".mp4":
            continue
        video_path = video_flie_path+"/"+video_name
        start_index = target_frame_count *  index
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        capture_interval = int(total_frames / target_frame_count)
        # 开始截取图像
        frame_count = 0
        while frame_count//capture_interval < target_frame_count:
            ret, frame = cap.read()
            if not ret:
                break  # 视频读取结束
            # 每隔一定帧数截取一帧
            if frame_count % capture_interval == 0:
                output_path = os.path.join(output_folder, f"{start_index + frame_count // capture_interval}.jpg")
                output_list.append((frame,output_path))
            frame_count += 1
        # 释放视频捕捉对象
        cap.release()
    return output_list
@count_time
def main():
    image_delete_local("image_goods")
    image_delete_local("image_pre")
    img_list = image_from_video(20)
    bg = cv2.imread("label/bg.jpg")
    bg_reisze_img= cv2.resize(bg ,(320,320))
    
    orign_resize_img_list= resize_list(img_list)

    cluster_center_img_list = []
    cluster_border_list =[]
    cluster_center_list = []

    for index,orign_resize_img in enumerate(orign_resize_img_list):

        # 寻找聚类中心
        cluster_centers = get_cluster_centers(orign_resize_img,show=False )
        # 获取聚类中心图像
        cluster_center_img,cluster_border = get_center_img(orign_resize_img,cluster_centers,show = False )
        cluster_center_img_list.append(cluster_center_img)
        cluster_border_list.append(cluster_border)
        cluster_center_list.append(cluster_centers)

    goods_index_list,goods_cluster_list = find_goods_centers(bg_reisze_img,cluster_center_img_list)
    goods_border = get_goods(bg_reisze_img,goods_index_list,orign_resize_img_list,cluster_center_list,cluster_border_list,is_show=False )
    for index,i in enumerate(goods_index_list):
        x_min,x_max,y_min,y_max = goods_border[index]
        good_img_path  = "image_goods"+ '/' + f"{index}" + ".jpg"
        cv2.imwrite(good_img_path,orign_resize_img_list[index][y_min:y_max,x_min:x_max])

if __name__ == "__main__":
    main()