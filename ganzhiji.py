from random import random
import numpy as np
weight = np.random.random(2)/1000
b_weight=np.random.random()/1000
# 生成随机权重
# weight = [0.01,0.02]
# b_weight =0.4
# 逻辑数据
simple_data =[[0,0],[0,1],[1,0],[1,1]]
expected_result = [0,1,1,1]
threshold = 0.5#激活函数阈值

print(weight,b_weight)
print('+++++++++')
for diedaicishu in range(5):#迭代次数
    correct_num =0
    for idx,x in enumerate(simple_data):#四个数据训练一遍
        w=np.array(weight)
        input=np.array(x)
        ac=np.dot(w,input)+(b_weight*1)#计算
        if ac > threshold: #判断类别
            result =1
        else:
            result = 0
        if result==expected_result[idx]:#判断是否准确，是否是误差项
            correct_num+=1
        print('result{},expect{}'.format(result,expected_result[idx]))
        new_weight=[]
        print('x:{}'.format(x))
        for i ,c in enumerate(x):#如果预测错误，更新权重
            new_weight.append(weight[i] + (expected_result[idx]-result)*c)
        b_weight=b_weight+((expected_result[idx]-result)*1)
        weight=np.array(new_weight)
        print(weight,b_weight)
    print('{} of 4,  di{}ci'.format(correct_num,diedaicishu))




