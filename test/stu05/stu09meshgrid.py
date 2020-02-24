import tensorflow as tf

y = tf.linspace(-2., 2, 5)
print(y)
x = tf.linspace(-2., 2, 5)
print(x)

point_x, point_y = tf.meshgrid(x, y)
print(point_x)
print(point_y)

points = tf.stack([point_x, point_y], axis=2)
print(points.shape)
print(points)

print(tf.reshape(points, [25, 2]))

print('------------------------------------------------')
import matplotlib.pyplot as plt


def func(x):
    # x: [b,2]
    z = tf.math.sin(x[..., 0]) + tf.math.sin(x[..., 1])
    return z


y = tf.linspace(0., 2 * 3.14, 500)
x = tf.linspace(0., 2 * 3.14, 500)
point_x, point_y = tf.meshgrid(x, y)

print('x',point_x.shape)
print(point_x)
print('y',point_y.shape)
print(point_y)
points = tf.stack([point_x, point_y], axis=2)
print('point:',points.shape)
print(points)
z = func(points)
print('z:',z.shape)

plt.figure('plot 2d func value')
plt.imshow(z,origin='lower',interpolation='none')
plt.colorbar()

plt.figure('plot 2d func contour')
plt.contour(point_x,point_y,z)
plt.colorbar()
plt.show()
