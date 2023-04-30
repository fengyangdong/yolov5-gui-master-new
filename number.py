
xy = []
end = ""
# 第一步，执行txt文件，把里面的信息提取出来
with open("D:\OneDrive\桌面/train3.txt","r+") as fp:
    for row in fp:
        rowlist = row.split(" ")
        """
        rowlist：
        第一个参数是标注信息
        第二个参数是横坐标
        第三个参数是纵坐标
        第四个和第五个参数是横纵高度
        """
        # 第二步，得到xy的坐标
        xy += [[int(float(rowlist[1])*10000), int(float(rowlist[2])*10000), int(float(rowlist[1])*10000 + float(rowlist[2])*10000)]]
        """
        xy = [[x1, y1, x1+y1], [x2, y2, x2+y2], …………]
        x1y1 = [x1, y1]
        """
# 第三步，删除点坐标，并且求出h的高度
# 1，删除左上点和右下的点，并且记录y值
xy = sorted(xy, key=(lambda x: x[2]))
x1 = xy[0][0]
x8 = xy[-1][0]
del xy[0]
del xy[-1]

x2 = xy[0][0]
x7 = xy[-1][0]
del xy[0]
del xy[-1]

xy = sorted(xy, key=(lambda x: x[1]))
x5 = xy[0][0]
x4 = xy[-1][0]
del xy[0]
del xy[-1]


x6 = xy[0][0]
x3 = xy[-1][0]
del xy[0]
del xy[-1]


print(f"x1 = {x1} x2 = {x2} x3 = {x3} x4 = {x4} \nx5 = {x5} x6 = {x6} x7 = {x7} x8 = {x8}")
h = ((x5 + x6 + x7 + x8) - (x1 + x2 + x3 + x4)) / 4
print("h的大小",h)
# 第四步，计算每个点上面对应的值
# 1，求出半球的间隙
hemisphere_long = h/18
print(f"半球长度为：{hemisphere_long}")
print(f"底下的x的坐标为{(x5 + x6 + x7 + x8)/4}")
# 2，遍历所有的剩下重要的点，按照x的顺序遍历
for x, y, l in xy:
    print(f"原始的x：{x}")
    print("此时的y", y)
    x = ((x5 + x6 + x7 + x8) / 4) - x
    print(x)
    if -1 * hemisphere_long <= x < hemisphere_long:
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
    elif 17*hemisphere_long <= x < 19 * hemisphere_long:
        end += "0"
    print(end)