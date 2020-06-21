import numpy as np

train_acc = [0.6292, 0.6342, 0.6304, 0.6467, 0.6285, 0.6212, 0.6291,0.6311, 0.6208]
test_acc = [0.6643, 0.6632, 0.6614, 0.6541, 0.6365, 0.6575, 0.6632, 0.6311, 0.6730]

train_num = [11831, 6053, 2332, 2015, 2105, 2273, 2842, 2635, 2223]
test_num = [5663, 2904, 1078, 983, 993,1054, 1333, 1255, 1058]


# #需要求加权平均值的数据列表
# elements = [1,1,1]
# #对应的权值列表
# weights = [1,2,3]

train_avg = np.average(train_acc, weights=train_num)
print(train_avg)

test_avg = np.average(test_acc, weights=test_num)
print(test_avg)