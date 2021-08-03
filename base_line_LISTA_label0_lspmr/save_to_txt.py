def save_to_txt(what_to_save, filename, mode='a'):
    """
    用法：
    1、首先确定想要保存的变量是一个矩阵，而不是一个3维以上的张量、也不是个list！！
        就在主函数的Console里打印出来就可以。
    2、Console里输入import save_to_txt
    3、save_to_txt.save_to_txt(变量名, "txt文件名")，注意此时这个txt文件不必存在，他可以自己创建。
    """
    file = open(filename, mode)
    for i in range(len(what_to_save)):
        file.write(str(what_to_save[i]) + '\n')
    file.close()