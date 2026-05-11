###### 自定义Layer

```python
class InnerProductLayer(Layer):
    """
      Input shape
        - a list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Output shape
        - 3D tensor with shape: ``(batch_size, N*(N-1)/2 ,1)`` if use reduce_sum. or 3D tensor with shape: ``(batch_size, N*(N-1)/2, embedding_size )`` if not use reduce_sum.
    """
    def __init__(self, reduce_sum=True, **kwargs):
        self.reduce_sum = reduce_sum
        super(InnerProductLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `InnerProductLayer` layer should be called '
                             'on a list of at least 2 inputs')
        pass

    def call(self, inputs, **kwargs):  # 输入如果为[Tensor], 则input_shape为[TensorShape]
        embed_list = inputs
        pass

    def compute_output_shape(self, input_shape):
        pass

    def get_config(self, ):
        config = {'reduce_sum': self.reduce_sum, }
        base_config = super(InnerProductLayer, self).get_config()
        base_config.update(config)
        return base_config
```

###### Tensor操作

```python
a = tf.random.normal(shape=(10, 5, 6))
b = tf.random.normal(shape = (6, 16))
tf.tensordot(a, b, axes=(-1,0)) #shape=(10, 5, 16)
```

```python
c = tf.random.normal(shape=(10, 8, 6))
querys = tf.stack(tf.split(c, 2, axis=2)) #shape=(2, 10, 8, 3)

tf.stack(tf.split(c, 2, axis=2),axis=1) # shape=(10, 2, 8, 3)
tf.stack(tf.split(c, 2, axis=1),axis=1) # shape=(10, 2, 4, 6)
tf.concat(tf.split(c, 2),axis=-1) # shape=(5, 8, 12)
tf.squeeze()#Removes dimensions of size 1 from the shape of a tensor.
```

```python
tf.transpose(c, [0, 2, 1]) # shape=(10, 6, 8)
tf.expand_dims(c, axis=0) # shape=(1, 10, 6, 8)
tf.expand_dims(c, axis=1) # shape=(10, 1, 6, 8)
```



###### 矩阵乘积

所有大于二维的，最终都是以二维为基础堆叠在一起，所以在矩阵运算的时候，其实最后都可以转成我们常见的二维矩阵运算，遵循的原则是：在多维矩阵相乘中，需最后两维满足shape匹配原则，最后两维才是有数据的矩阵，前面的维度只是矩阵的排列而已。前面的维度要满足broadcast才行，就是要么有一个维度为1，要么维度相等

```python
a = tf.random.normal(shape=(10,1, 5, 6))
b = tf.random.normal(shape=(1, 4, 6, 5))
tf.matmul(a, b) # shape=(10, 4, 5, 5)
```

###### 聚合操作

```python
a = tf.random.normal(shape=(10,2, 5, 6))
tf.reduce_sum(a,axis=-1,keepdims=True) # shape=(10, 2, 5, 1)
tf.reduce_sum(a, axis=-1) # shape=(10, 2, 5)
tf.reduce_sum(c,axis=1) # shape=(10, 5, 6)
```

###### Tensor运算

```python
a = tf.random.normal(shape=(10, 5, 6, 1))
b = tf.random.normal(shape=(5, 1, 6))
tf.multiply(a,b) # shape=(10, 5, 6, 6)
```

```python
x_0 = tf.expand_dims(inputs, axis=2) #batch * dim * 1
x_l = x_0
for i in range(self.layer_num):
    if self.parameterization == 'vector':
        # xl_w = batch*1*1
        xl_w = tf.tensordot(x_l, self.kernels[i], axes=(1, 0))
        # batch * dim *1 
        dot_ = tf.matmul(x_0, xl_w)
        # batch * dim *1
        x_l = dot_ + self.bias[i] + x_l
x_l = tf.squeeze(x_l, axis=2)
```
###### 模型夹加载
```python
import tensorflow as tf
import tensorflow.keras as tf
....
tf.saved_model.save(model, 'saved' + '/' + model_name)
model = keras.models.load_model('saved/egg_open_dcnmix')
# tf2.4可以正常load_model，但是tf1.15load_model报错。
```
