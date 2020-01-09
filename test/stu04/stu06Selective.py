import tensorflow as tf

a = tf.random.uniform([4,35,8])
# a中第0维的 2，1，4，0
print(tf.gather(a,axis=0,indices=[2,1,4,0]))

aa = tf.gather(a,axis=1,indices=[5,8,10])
aaa = tf.gather(aa,axis=2,indices=[1,3,5])
print(aaa)

# a[0]
tf.gather_nd(a,[0])
# a[0,1]
tf.gather_nd(a,[0,1])
tf.gather_nd(a,[0,1,2])
# [a[0,1,20]]
tf.gather_nd(a,[[0,1,2]])

# （算里面再拼接）
# a[0,0],a[1,1]合在一起  最终[2,8]
tf.gather_nd(a,[[0,0],[1,1]])


a = tf.random.uniform([4,28,28,3])

# 取出对应维 是 True的值
tf.boolean_mask(a,mask=[True,True,False,False])

tf.boolean_mask(a,mask=[True,True,True],axis = 3)

# 可以多维
a = tf.ones([2,3,4])
# 对应[2,3]的位置上 取出  3个True最终[3,4]
tf.boolean_mask(a,mask=[[True,False,False],[False,True,True]])