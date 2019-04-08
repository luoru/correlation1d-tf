# Correlation1d Tensorflow version
DispNet correlation1d layer tensorflow implementation. the code has been tested success! This code mainly based on
(https://github.com/sampepose/flownet2-tf.git), which has implement the correlation of flownet. Other version of correlation1d in github still have some bugs
# Compiling
just run compille.sh

#### *correlation1d example*
```shell
$ python
```
```python
>>> import tensorflow as tf
>>> x1 = tf.random_normal([1, 10, 10, 4])
>>> x2 = tf.random_normal([1, 10, 10, 4])
>>> conv_x1 = tf.layers.conv2d(x1, 10, 3, 1, 'same', name='conv_x1')
>>> conv_x2 = tf.layers.conv2d(x2, 10, 3, 1, 'same', name='conv_x2')
>>> corr = correlation1d(conv_x1, conv_x1, 1, 4, 1, 1, 4)
>>> grad_corr2x1 = tf.gradients(corr, x1)
```
