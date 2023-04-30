import torch
import time
import numpy as np
import math
import numpy as np

def get_distance_from_point_to_line(point, line_point1, line_point2):
    #对于两点坐标为同一点时,返回点与点的距离
    if line_point1 == line_point2:
        point_array = np.array(point)
        point1_array = np.array(line_point1)
        return np.linalg.norm(point_array -point1_array )
    #计算直线的三个参数
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
        (line_point2[0] - line_point1[0]) * line_point1[1]
    #根据点到直线的距离公式计算距离
    distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))
    return distance

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


data = ""
datas = {}

# 第一步，执行txt文件，把里面的信息提取出来
def numder(tensor,gn):
    save_conf = False
    xy = []
    # print(tensor, gn)
    end  =''

    for *xyxy, conf, cls in reversed(tensor):
        print(len(tensor))

        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        row = ('%g ' * len(line)).rstrip() % line
        rowlist = row.split(" ")
        """
        rowlist：
        第一个参数是标注信息
        第二个参数是横坐标
        第三个参数是纵坐标
        第四个和第五个参数是横纵高度
        """
        # 第二步，得到xy的坐标
        xy += [[int(float(rowlist[1]) * 10000), int(float(rowlist[2]) * 10000), int(float(rowlist[1]) * 10000 + float(rowlist[2]) * 10000)]]
        """
        xy = [[x1, y1, x1+y1], [x2, y2, x2+y2], …………]
        x1y1 = [x1, y1]
        """
        print("得到xy坐标")
        print(len(xy), xy)
    print(5)
    # 第三步，删除点坐标，并且求出h的高度
    # 0,删除多余的点
    xy = sorted(xy, key=(lambda x: x[2]))
    x1 = xy[0][0]
    y1 = xy[0][1]
    x2 =  xy[1][0]
    y2 = xy[1][1]
    y = (y1 +y2)/2
    x = (x1+x2)/2
    xy = sorted(xy, key=(lambda x: x[0]))
    temp = 0
    for i in xy:
        if i[0] < x1:
            temp +=1
        if temp >= 3:
            temp = -1
            break

    # xy = sorted(xy, key=(lambda x: x[0]), reverse=True)
    # for i in xy:
    #     temp = False
    #     if i[1] > (y - _high) and i[1] < (y + _high) and temp == False:
    #         x5 = i[0]
    #         y5 = i[1]
    #         temp = True
    #     if i[1] > (y - _high) and i[1] < (y + _high) and temp == True:
    #         x6 = i[0]
    #         y6 = i[1]
    #         break
    # xy = sorted(xy, key=(lambda x: x[0]))

    while len(xy)>20:
        if temp == -1:
            xy = sorted(xy, key=(lambda x: x[0]))
        else:
            xy = sorted(xy, key=(lambda x: x[0]), reverse=True)
        print("删除值：", xy[0])
        del xy[0]

    # 1，删除左上点和右下的点，并且记录y值
    print("删除完成，剩下的点")
    print(len(xy), xy)

    xy = sorted(xy, key=(lambda x: x[2]))
    x1 = xy[0][0]
    y1 = xy[0][1]
    x8 = xy[-1][0]
    y8 = xy[-1][1]
    del xy[0]
    del xy[-1]
    print("删除1,8")
    x2 =  xy[1][0]
    y2 = xy[1][1]
    x7 = xy[-1][0]
    y7 = xy[-1][1]
    del xy[0]
    del xy[-1]
    print("删除2,7")
    xy = sorted(xy, key=(lambda x: x[0]))
    x3 = xy[0][0]
    y3 = xy[0][1]
    x6 = xy[-1][0]
    y6 = xy[-1][1]
    del xy[0]
    del xy[-1]
    print("删除4,5")
    x4 = xy[0][0]
    y4 = xy[0][1]
    x5 = xy[-1][0]
    y5 = xy[-1][1]
    del xy[0]
    del xy[-1]
    print("删除3,6")



    # h = math.sqrt(((x7+x8+x5+x6-x3-x4-x2-x1)/4)**2+((y7+y8+y5+y6-y3-y4-y2-y1)/4)**2)
    h = (y3+y4+y8+y7-y1-y2-y5-y6)/4
    print("h", h)

    # 第四步，计算每个点上面对应的值
    # 1，求出半球的间隙
    hemisphere_long = h/18
    print("hemisphere_long", hemisphere_long)
    # print(f"半球长度为：{hemisphere_long}")
    # print(f"底下的x的坐标为{(x5 + x6 + x7 + x8)/4}")
    # 2，遍历所有的剩下重要的点，按照x的顺序遍历
    # line1 = (((x5+x6)/2), ((y5+y6)/2))
    # line2 = (((x8+x7)/2), ((y7+y8)/2))
    line1 = (((x1+x2)/2), ((y1+y2)/2))
    line2 = (((x5+x6)/2), ((y5+y6)/2))
    xy = sorted(xy, key=(lambda x: x[0]))
    print("开始点操作")
    print("剩余点", len(xy), "\n", xy)
    for x, y, l in xy:
        # print(f"原始的x：{x}")
        # print("此时的y", y)
        point = (x, y)
        print("点进入")
        x = get_distance_from_point_to_line(point, line1,line2)
        print("点出来")
        if -1*hemisphere_long<= x < hemisphere_long:
            end += "9"
        elif hemisphere_long <= x < 3 * hemisphere_long:
            end += "8"
        elif 3*hemisphere_long <= x < 5 * hemisphere_long:
            end += "7"
        elif 5*hemisphere_long <= x < 7 * hemisphere_long:
            end += "6"
        elif 7*hemisphere_long <= x < 9 * hemisphere_long:
            end += "5"
        elif 9*hemisphere_long <= x < 11 * hemisphere_long:
            end += "4"
        elif 11*hemisphere_long <= x < 13 * hemisphere_long:
            end += "3"
        elif 13*hemisphere_long <= x < 15 * hemisphere_long:
            end += "2"
        elif 15*hemisphere_long <= x < 17 * hemisphere_long:
            end += "1"
        elif 17*hemisphere_long <= x < 19*hemisphere_long:
            end += "0"
        else:
            print("此时没有符合的")
            return [-1, 0]
        print(end, x)
    global data
    global datas
    if len(end) == 12:
        if end not in datas.keys():
            datas[end] = 1
            print("新数据，此时的datas为：", datas)
        else:
            datas[end] += 1
            print("旧数据，此时的datas为：", datas)
            if datas[end] >= 10:
                if data == "":
                    print("data为空，第一次执行")
                    # datas[end] = 1
                    data = end
                    return [1, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), end, datas[end], time.time()]
                elif data == end:
                    print("此时的data和end一样")
                    data = end
                    return [0, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), end, datas[end], time.time()]
                elif data != end:
                    print("此时有一个新数据")
                    datas.clear()
                    datas[end] = 10
                    # datas = {end, datas[end]}
                    data = end
                    return [1, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), end, 100, time.time()]
    print("长度 ", datas[end])
    return [-1,0]




