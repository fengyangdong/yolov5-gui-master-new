import time
import os

def txt(number):
    print("现在", number)
    path_root = os.path.dirname(__file__) + "\save"
    path_year = path_root+'\\'+time.strftime("%Y", time.localtime())
    folder = os.path.exists(path_year)
    if not folder:
        os.makedirs(path_year)
    path_data = path_year +'\\'+time.strftime("%m-%d", time.localtime())

    with open(f"{path_data}.txt", "a") as fp:
        fp.write(number)
        fp.write("\n")

