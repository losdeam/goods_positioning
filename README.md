## 前提条件：
使用当前版本的定位模块的前提条件如下

1. 物体与背景需要具有较大的差异
2. 图像中仅运行同时出现单个物体
3. 背景应尽可能的单调
## 环境配置：
语言环境: python3.10<br />环境安装：
```powershell
pip install -r requirements.txt
```

## 使用方法:

1. 在video中放入物体的视频,并label中放入视频的背景，
2. 运行main.py即可直接在imag_goods中获取到物体的图片,同时也可以在label文件夹下获取到yolo框架所需的训练数据的标签。
## 设计思路：
### 获取相同大小的图像
将图像经由resize转换为相同大小后再进行操作
### 寻找特征集中区域
使用SIFT算法来获取图像中的关键点，然后使用k-means聚类算法来获取聚类中心。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/40362764/1704263211402-5b957f42-3290-4e36-838f-cec15bf51ed8.png#averageHue=%23a79e97&clientId=udb46ac61-aca3-4&from=paste&height=463&id=ue9ac2aeb&originHeight=434&originWidth=801&originalType=binary&ratio=0.9375&rotation=0&showTitle=false&size=204818&status=done&style=none&taskId=u9a0ca2a0-5754-4195-9358-2e5ca658bc1&title=&width=854.4)
### 根据聚类中心来截取一定大小的图像
对每个聚类中心获取一定一定大小的图像，同时进行一次边界判断，由于本模块使用ssim（结构相似度）来判断图像之间的相似度，所以需要保证图像大小相同，为了方便后续计算，则将越界部分由未越界的一侧相同大小部分进行填充。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/40362764/1704263593934-b1309b50-4153-46f3-bd0f-9bde20af9483.png#averageHue=%2372811f&clientId=udb46ac61-aca3-4&from=paste&height=469&id=ua1ed6cbe&originHeight=440&originWidth=567&originalType=binary&ratio=0.9375&rotation=0&showTitle=false&size=200723&status=done&style=none&taskId=u01f6c0d3-149d-4155-b1b6-4ad61254d1a&title=&width=604.8)

### 寻找位于物体上的聚类中心
将背景图转换为与获取的聚类中心片段相同大小。然后获取所有聚类中心中与背景相似度最低的聚类中心作为物体的聚类中心

### 以物体聚类中心为核心进行二分定位
 二分的策略如下<br />![](https://cdn.nlark.com/yuque/0/2024/jpeg/40362764/1704178396518-a68aa3b7-dfb4-4bec-91fe-475ef7a91c06.jpeg)

1. 判断红色区域与背景相同区域的相似度，设为similarity_r
2. 若是 similarity_r> 特定值,则表示红色区域内存在物体内容，那么蓝色区域中必定也存在物体内容，设置left为mid，进行下一次循环
3. 否则，表示红色区域内不存在物体内容，将right设置为mid，进行下一次循环

![二分过程.gif](https://cdn.nlark.com/yuque/0/2024/gif/40362764/1704265593997-f5345751-6f9e-4639-afb1-c87533102605.gif#averageHue=%2324d735&clientId=udb46ac61-aca3-4&from=drop&id=u010f9bf7&originHeight=440&originWidth=804&originalType=binary&ratio=0.9375&rotation=0&showTitle=false&size=417344&status=done&style=none&taskId=udb575b92-1542-45ce-bead-1200b3e6d21&title=)<br />以这样的方式进行四次二分查找即可确认物体的四个边界。至此，定位功能正式完成。

### <br />

