#!/usr/bin/env python
# encoding: utf-8

# 导入科学计算包numpy和运算符模块operator
from numpy import *
import operator
import matplotlib.pyplot as plt


def create_data_set():
    """
    创建数据集和标签

     调用方式
     import kNN
     group, labels = kNN.createDataSet()
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify_0(inX, data_set, labels, k):
    """
    inx[1,2,3]
    DS=[[1,2,3],[1,2,0]]
    inX: 用于分类的输入向量
    data_set: 输入的训练样本集
    labels: 标签向量
    k: 选择最近邻居的数目
    注意：labels元素数目和dataSet行数相同；程序使用欧式距离公式.

    预测数据所在分类可在输入下列命令
    kNN.classify0([0,0], group, labels, 3)
    """
    # 1. 距离计算
    data_set_size = data_set.shape[0]  # shape使用方法参考readme
    print('data_set_size:', data_set_size)
    # tile生成和训练样本对应的矩阵，并与训练样本求差
    """
    tile: 列-3表示复制的行数， 行-1／2表示对inx的重复的次数
    inx = [1, 2, 3]
    In [8]: tile(inx, (3, 1))
    Out[8]: array([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]])

    In [9]: tile(inx, (3, 2))
    Out[9]:
    array([[1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3]])
    """
    print('tile:', tile(inX, (data_set_size, 1)))
    diff_mat = tile(inX, (data_set_size, 1)) - data_set  # 每一行相减
    """
    欧氏距离： 点到点之间的距离
       第一行： 同一个点 到 data_set的第一个点的距离。
       第二行： 同一个点 到 data_set的第二个点的距离。
       ...
       第N行： 同一个点 到 data_set的第N个点的距离。

    [[1,2,3],[1,2,3]]-[[1,2,3],[1,2,0]]
    (A1-A2)^2+(B1-B2)^2+(c1-c2)^2
    """
    # 取平方
    sq_diff_mat = diff_mat ** 2  # 矩阵乘方
    # 将矩阵的每一行相加
    sq_distances = sq_diff_mat.sum(axis=1)
    # 开方
    distances = sq_distances ** 0.5
    # 根据距离排序从小到大的排序，返回对应的索引位置
    # argsort() 是将x中的元素从小到大排列，提取其对应的index（索引），然后输出到y。
    # 例如：y=array([3,0,2,1,4,5]) 则，x[3]=-1最小，所以y[0]=3,x[5]=9最大，所以y[5]=5。
    # print 'distances=', distances
    print('distances:', distances)
    sorted_dist_indicies = distances.argsort()
    print("souted_dist_indicies:", sorted_dist_indicies)

    # 2. 选择距离最小的k个点
    class_count = {}  # 字典作为保存的容器 label为key count为value
    for i in range(k):
        # 找到该样本的类型
        vote_label = labels[sorted_dist_indicies[i]]
        # 在字典中将该类型加一
        # 字典的get方法
        # 如：list.get(k,d) 其中 get相当于一条if...else...语句,参数k在字典中，字典将返回list[k];如果参数k不在字典中则返回参数d,如果K在字典中则返回k对应的value值
        # l = {5:2,3:4}
        # print l.get(3,0)返回的值是4；
        # Print l.get（1,0）返回值是0；
        print('vote_label:', vote_label)
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # 3. 排序并返回出现最多的那个类型
    # 字典的 items() 方法，以列表返回可遍历的(键，值)元组数组。
    # 例如：dict = {'Name': 'Zara', 'Age': 7}   print "Value : %s" %  dict.items()   Value : [('Age', 7), ('Name', 'Zara')]
    # sorted 中的第2个参数 key=operator.itemgetter(1) 这个参数的意思是先比较第几个元素
    # 例如：a=[('b',2),('a',1),('c',0)]  b=sorted(a,key=operator.itemgetter(1)) >>>b=[('c',0),('a',1),('b',2)] 可以看到排序是按照后边的0,1,2进行排序的，而不是a,b,c
    # b=sorted(a,key=operator.itemgetter(0)) >>>b=[('a',1),('b',2),('c',0)] 这次比较的是前边的a,b,c而不是0,1,2
    # b=sorted(a,key=opertator.itemgetter(1,0)) >>>b=[('c',0),('a',1),('b',2)] 这个是先比较第2个元素，然后对第一个元素进行排序，形成多级排序。
    print('class_count:', class_count)
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def test1():
    group, labels = create_data_set()
    classify_0([0, 0], group, labels, 3)


# ------------------------------------------------------------------

def file2matrix(filename):
    """
    导入训练数据
    :param filename: 数据文件路径
    :return: 数据矩阵return_mat和对应的类别class_label_vector
    """
    with open(filename) as fr:
        array_lines = fr.readlines()
        # 获得文件中的数据行的行数
        num_lines = len(array_lines)
        # 生成对应的空矩阵
        # zeros 函数说明 a new array of given shape and type, filled with zeros.
        # 例如: zeros(2,3)就是生成一个2*3的矩阵,各个位置全是0
        return_mat = zeros((num_lines, 3))  # prepare matrix to return
        class_label_vector = []  # prepare labels return
        index = 0
        for line in array_lines:
            # 返回移除字符串头尾指定的字符生成新的字符串,过滤回车
            line = line.strip()
            # 以制表符切割字符串
            list_from_line = line.split('\t')
            # 每列的属性数据
            return_mat[index, :] = list_from_line[0:3]
            # 每列的类别数据 就是label标签数据
            class_label_vector.append(int(list_from_line[-1]))
            index += 1
        # 返回数据矩阵return_mat和对应的类别的class_label_vector
        return return_mat, class_label_vector


def auto_norm(data_set):
    """
    归一化特征值, 消除属性之间量级不同导致的影响
    :param data_set: 数据集
    :return:
    归一化后的数据集normalDataSet, ranges和minVals即最小值与范围

    归一化公式:
        Y = (X - Xmin)/ (Xmax - Xmin)
        其中的min和max
        分别为数据集中的最小特征值和最大特征值.
        该函数可以自动将数字特征值转化为0到1的区间
    """
    # 计算每种属性的最大值,最小值,范围
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    # 极差
    ranges = max_vals - min_vals
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    # 生成与最小值之差组成的矩阵
    norm_data_set = data_set - tile(min_vals, (m, 1))
    # 将最小值之差除以范围组成矩阵
    norm_data_set = norm_data_set / tile(ranges, (m, 1))  # element wise divide
    return norm_data_set, ranges, min_vals


def dating_class_test():
    """
    对约会网站的测试方法
    :return: 错误数
    """
    # 设置测试数据的一个比例(训练数据集比例 = 1- ho_ratio)
    ho_ratio = 0.1  # 测试范围, 一部分测试一部分作为样本
    # 从文件中加载数据
    dating_mat, dating_labels = file2matrix("datingTestSet.txt")
    # 归一化数据
    norm_mat, ranges, min_vals = auto_norm(dating_mat)
    # m 表示数据的行数, 即矩阵的第一维
    m = norm_mat.shape[0]
    # 设置测试的样本数量, num_test_vecs:m表示训练向本的数量
    num_test_vecs = int(m * ho_ratio)
    print('num_test_vecs:', num_test_vecs)
    error_count = 0.0
    for i in range(num_test_vecs):
        # 对数据测试
        result = classify_0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (result, dating_labels[i]))
        if result != dating_labels[i] : error_count += 1.0
    print("the total error rate is: %f" % (error_count / float(num_test_vecs)))
    print("error:", error_count)


def test2():
    dating_mat, dating_label = file2matrix('datingTestSet.txt')

    norm_set, ranges, min_vals = auto_norm(dating_mat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(norm_set[:, 0], norm_set[:, 1],
               15.0 * array(dating_label), 15.0 * array(dating_label))
    plt.show()


if __name__ == '__main__':
    # test1()
    # print(file2matrix('datingTestSet.txt'))
    # dating_mat, dating_label = file2matrix('datingTestSet.txt')
    # print('mat:', dating_mat)
    # print('label:', dating_label[0:20])
    # norm_set, ranges, min_val = auto_norm(dating_mat)
    # print('normal_set:', norm_set)
    # print('ranges:', ranges)
    # print("min_vals:", min_val)
    # test2()
    dating_class_test()
    pass
