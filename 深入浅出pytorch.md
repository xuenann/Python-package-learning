## 第二章

### 1 pytorch包的结构

#### 主要模块

1. ==torch模块==

   > 包含pytorch经常使用的一些激活函数：Sigmoid（`torch.sigmoid`）、ReLU（`torch.relu`）、Tanh（`torch.tanh`）
   >
   > 以及一些pytorch张量的一些操作：矩阵的乘法（`torch.mm`）、张量元素的选择（`torch.select`）
   >
   > 还有一类函数可以产生一定形状的张量：产生全0的张量（`torch.zeros`）

2. ==torch.Tensor模块==

   > 定义了torch中的张量类型，其中张量有不同的数值类型（单精度、双精度浮点、整数类型等），而且张量有一定的维数和形状
   >
   > 张量的类中也包含着一系列的方法，返回新的张量或更改当前的张量：Tensor.add返回新的张量；tensor.add_更改当前张量的值

3. ==torch.sparse模块==

   > 定义了稀疏张量，其中构造的稀疏张量采用的是COO格式，主要方法使用一个长整型定义非零元素的位置，用浮点数张量定义对应非零元素的值，
   >
   > 稀疏张量之间可以做元素的加、减、乘、除运算和矩阵乘法

4. ==torch.cuda模块==

   > 定义了与CUDA运算相关的一系列函数，包括但不限于检查系统的CUDA是否可用，当前进程对应的GPU序号，清除GPU上的缓存等

5. ==torch.nn模块==

   > 是pytorch神经网络模块化的核心，定义了一系列的模块，包括卷积层（`nn.ConvND`(N=123)）和线性层（全连接层）（`nn.Linear`）等
   >
   > 当构建深度学习模型的时候，`可以通过继承nn.Module`类并重写forward方法来实现一个新的神经网络
   >
   > 也定义了一系列损失函数：平方损失函数（`torch.nn.MSELoss`）、交叉熵损失函数（`torch.nn.CrossEntropyLoss`）等

6. ==torch.nn.functional函数模块==

   > 定义了一些核神经网络相关的函数，
   >
   > 还定义了一些不常用的激活函数，：`torch.nn.functional.relu6`、`torch.nn.functional.elu`

7. ==torch.nn.init模块==

   > 定义了神经网络权重的初始化，
   >
   > 为了解决神经网络权重的初始化问题，使用了很多初始化方法：均匀初始化（`troch.nn.init.uniform_`）、正态分布归一化（`torch.nn.init.normal_`）

8. ==torch.optim模块==

   > 定义了一系列优化器：随机梯度下降算法（`torch.optim.SGD`）、AdaGrad算法（`torch.optim.Adagrad`）、RMSProp算法（`torch.optim.RMSprop`）、Adam算法（`torch.optim.Adam`）
   >
   > 还包含学习率衰减的算法：`torch.optim.lr_scheduler`;
   >
   > 学习率阶梯下降算法（`torch.optim.lr_scheduler.StepLR`）、余弦退火算法（`torch.optim.lr_scheduler.CosineAnnealingLR`）

9. ==torch.autograd模块==

   > 自动微分算法模块，定义了一系列自动微分函数
   >
   > `torch.autograd.backward`:主要用于在求得损失函数之后进行反向梯度传播
   >
   > `torch.autogard.grad`:用于一个标量张量（即只有一个分量的张量）对另一个张量求导，以及在代码中设置不参与求导的部分
   >
   > 还内置了数值梯度功能和检查自动微分引擎是否输出正确结果的功能

10. ==torch.distributed模块==

    > 分布式计算模块；主要功能是提供pytorch并行运行环境，其主要支持的后端有MPI、Gloo、NCCL

11. ==torch.distributions模块==

    > 提供一系列类，是的pytorch能够对不同的分布进行采样，并且生成概率采样过程的计算图

12. ==torch.hub模块==

    > 提供一系列预训练的模型供用户使用，
    >
    > `torch.hub.list`:获取某个模型镜像站点的模型信息
    >
    > `torch.hub.load`:载入预训练的模型，载入后的模型可以保存到本地

13. ==torch.jit模块==

    > 即时编辑器模块，把pytorch的动态图转换成可以优化和序列化的静态图；
    >
    > 主要的工作原理是通过输入预先定义好的张量，追踪整个动态图的构建过程，得到最终的动态图后转换为静态图

14. ==torch.multiprocessing模块==

    > 定义了多进程API，通过使用这个模块，可以启动不同的进程，每个进程运行不同的深度学习模型，并且能够在进程间共享张量（通过共享内存的方式）

15. ==torch.random模块==

    > 提供了一系列的方法来保存和设置随机数生成器的状态，包括使用

16. ==torch.onnx模块==

    > 定义了pytorch导出和载入ONNK格式的深度学习模型描述文件
    >
    > 引入这个模块可以方便pytorch导出模型给其他深度学习框架使用，或者让pytorch可以载入其他深度学习框架构建的深度学习模型



#### 辅助工具模块

1. ==torch.utils.bottleneck模块==

   > 可以用来检查深度学习模型中模块的运行时间；从而可以找到导致性能瓶颈的模块，进一步优化该模块

2. ==torch.utils.checkpoint模块==

   > 节约深度学习使用的内存
   >
   > 通过这个模块记录中间数据的计算过程，然后丢弃这些数据，需要用到时再重新计算（以计算时间换内存空间）

3. ==torch.utils.cpp_extension模块==

   > 定义了pytorch的C++扩展，主要包含两个类：
   >
   > CppExtension定义了使用C++来编写的扩展模块的源代码相关信息
   >
   > CUDAExtension定义了C++/CUDA编写的扩展模块的源代码相关信息

4. ==torch.utils.data模块==

   > 引入了数据集（Dataset）和数据载入器（DataLoader）的概念；
   >
   > 前者代表包含了所有数据的数据集，通过索引能够得到某一条特定的数据
   >
   > 后者通过对数据集的包装，可以对数据集进行随机排列（Shuffle）和采样（Sample），得到一系列打乱数据顺序的迷你批次

5. ==torch.utils.dlpacl模块==

   > 定义了pytorch张量和DLPack张量存储格式之间的转换，用于不同框架之间张量数据的交换

6. ==torch.utils.tensorboard模块==

   > 对TensorBoard数据可视化工具的支持



### 2 张量的创建和维度的操作

#### 张量的数据类型

| 数据类型       | pytorch类型                     | CPU上的张量        | GPU上的张量            |
| -------------- | ------------------------------- | ------------------ | ---------------------- |
| 32位浮点数     | torch.float32<br />torch.float  | torch.FloatTensor  | torch.cuda.FloatTensor |
| 64位浮点数     | troch.float64<br />torch.double | torch.DoubleTensor | torch.cuda.DoubleTensor     |
| 16位浮点数     | torch.float16<br />torch.half   | torch.HalfTensor   | torch.cuda.HalfTensor       |
| 8位无符号整数  | torch.uint8                     | torch.ByteTensor   | torch.cuda.ByteTensor       |
| 8位带符号整数  | torch.int8                      | torch.CharTensor   | torch.cuda.CharTensor       |
| 16位带符号整数 | torch.int16<br />torch.short    | troch.ShortTensor  | troch.cuda.ShortTensor      |
| 32位带符号整数 | torch.int32<br />torch.int      | torch.IntTensor    | torch.cuda.IntTensor        |
| 64位带符号整数 | torch.int64<br />torch.long     | torch.LongTensor   | torch.cuda.LongTensor       |
| 布尔型         | torch.bool                      | torch.BoolTensor   | torch.cuda.BoolTensor       |



#### 张量的创建方式

1. torch.tensor创建

   > 通过torch.tensor函数创建；预先有数据（包括列表和Numpy数组），可以通过这个方法转换

2. pytorch内置的函数创建张量

   > 通过指定张量的形状，返回给定形状的张量。
   >
   > `torch.rand(3,3)`：创建3*3矩阵，服从[0,1)上的均匀分布
   >
   > `torch.randn()`：服从标准正态分布
   >
   > `torch.zeros()`、`torch.ones()`、`torch.eye()`：单位矩阵
   >
   > `torch.randint(0,10,(3,3))`：生成[0,10)之间均匀分布整数的3*3矩阵

3. 通过已知张量创建形状相同的张量

   > ```python
   > t=torch.randn(3*3)
   > torch.zeros_like(t)		#生成一个元素全为0的张量，形状与t相同
   > torch.ones_like(t)
   > torch.rand_like(t)
   > torch.randn_like(t)
   > ```

4. 通过已知张量创建形状不同但数据类型相同的张量

   > ```
   > t.new_tensor([1,2,3]).dtype		#torch.float32
   > t.new_zeros(3,3)		#生成相同类型且元素全为0的张量
   > t.new_ones(3,3)
   > ```



#### 张量的存储设备

分为CPU和GPU

```
torch.randn(3,3,device="cpu")		#获取存储在CPU上的一个张量
torch.randn(3,3,device="cuda:0")	#获取存储在0号GPU上的一个张量
torch.randn(3,3,device="cuda:1").device	#获取当前张量的设备

torch.randn(3,3,device="cuda:0").cpu()	#张量从0号GPU转移到CPU

torch.randn(3,3,device="cuda:0").cuda(1)	#张量从0号GPU转移到1号GPU
torch.randn(3,3,device="cuda:0").to("cuda:1")
```



####  张量维度相关方法

```python
t = torch.randn(3,4,5) # 产生一个3×4×5的张量

t.ndimension() # 获取维度的数目
t.nelement() # 获取该张量的总元素数目

t.size() # 获取该张量每个维度的大小，调用方法
t.shape # 获取该张量每个维度的大小，访问属性
t.size(0) # 获取该张量维度0的大小，调用方法

t = torch.randn(12) # 产生大小为12的向量

t.view(3, 4) # 向量改变形状为3×4的矩阵
t.view(4, 3) # 向量改变形状为4×3的矩阵
t.view(-1, 4) # 第一个维度为-1，PyTorch会自动计算该维度的具体值
				# view方法不改变底层数据，改变view后张量会改变原来的张量

t.data_ptr() # 获取张量的数据指针
t.view(3,4).data_ptr() # 数据指针不改变
t.view(4,3).data_ptr() # 同上，不改变
t.view(3,4).contiguous().data_ptr() # 同上，不改变
t.view(4,3).contiguous().data_ptr() # 同上，不改变

t.view(3,4).transpose(0,1).data_ptr() # transpose方法交换两个维度的步长
t.view(3,4).transpose(0,1).contiguous().data_ptr() # 步长和维度不兼容，重新生成张量
```

* 改变张量形状的方法
  * view方法：不改变张量底层的数据
  * reshape方法：会在形状信息不兼容的时候自动生成一个新的张量，并自动复制原始张量的数据



#### 张量的索引和切片

```python
t = torch.randn(2,3,4) # 构造2×3×4的张量

t[1,2,3] # 取张量在0维1号、1维2号、2维3号的元素（编号从0开始）
t[:,1:-1,1:3] # 仅仅一个冒号表示取所有的，-1表示最后一个元素

t[1,2,3] = -10	 # 直接更改索引和切片会更改原始张量的值
t > 0 # 张量大于零部分的掩码
t[t>0] # 根据掩码选择张量的元素，注意最后选出来的是一个向量
```



### 3 张量的运算

#### 单个张量的函数运算

```python
t1 = torch.rand(3, 4) # 产生一个3×4的张量

t1.sqrt() # 张量的平方根，张量内部方法
torch.sqrt(t1) # 张量的平方根，函数形式
				# 前两个操作不改变张量的值
    
t1.sqrt_() # 平方根原地操作，改变张量的值

torch.sum(t1) # 默认对所有的元素求和
torch.sum(t1, 0) # 对第0维的元素求和
torch.sum(t1, [0,1]) # 对第0、1维的元素求和

t1.mean() # 对所有元素求平均，也可以用torch.mean函数
t1.mean(0) # 对第0维的元素求平均
torch.mean(t1, [0,1]) # 对第0、1维元素求平均
```



#### 多个张量的函数运算

```python
t1 = torch.rand(2, 3)
t2 = torch.rand(2, 3)
t1.add(t2) # 四则运算，不改变参与运算的张量的值
t1+t2
t1.sub(t2)
t1-t2
t1.mul(t2)
t1*t2
t1.div(t2)
t1/t2
t1
t1.add_(t2) # 四则运算，改变参与运算张量的值
```



#### 张量的极值和排序

```
t = torch.randn(3,4) # 建立一个3×4的张量

torch.argmax(t, 0) # 函数调用，返回的是沿着第0个维度，极大值所在位置
t.argmin(1) # 内置方法调用，返回的是沿着第1个维度，极小值所在的位置

torch.max(t, -1) # 函数调用，返回的是沿着最后一个维度，包含极大值和极大值所在位置的元组
t.min(0) # 内置方法调用，返回的是沿着第0个维度，包含极小值和极小值所在位置的元组

t.sort(-1) # 沿着最后一个维度排序，返回排序后的张量和张量元素在该维度的原始位置
```



#### 矩阵的乘法和张量的缩进

```
a = torch.randn(3,4) # 建立一个3×4的张量
b = torch.randn(4,3) # 建立一个4×3的张量

torch.mm(a,b) # 矩阵乘法，调用函数，返回3×3的矩阵乘积
a.mm(b) # 矩阵乘法，内置方法
a@b # 矩阵乘法，@运算符号

a = torch.randn(2,3,4) # 建立一个大小为2×3×4的张量
b = torch.randn(2,4,3) # 建立一个张量，大小为2×4×3
torch.bmm(a,b) # （迷你）批次矩阵乘法，返回结果为2×3×3，函数形式
a.bmm(b) # 同上乘法，内置方法形式
a@b # 运算符号形式，根据输入张量的形状决定调用批次矩阵乘法
```

```
a = torch.randn(2,3,4) # 随机产生张量
b = torch.randn(2,4,3)

a.bmm(b) # 批次矩阵乘法的结果2×3×3
torch.einsum("bnk,bkl->bnl", a, b) # einsum函数的结果，和前面的结果一致
```



#### 张量的拼接和分割

1. `torch.stack`：通过传入的张量列表，同时指定并创建一个维度，把列表的张量沿着该维度堆叠起来，并返回堆叠后的张量；传入的张量列表中所有张量大小必须一致

2. `torch.cat`：通过传入的张量列表指定某一个维度，把列表中的张量沿着该维度堆叠起来，并返回堆叠以后的张量；传入的张量列表的所有张量除了指定堆叠的维度外，其他的维度大小必须一致

3. `torch.split`：输出张量沿着某个维度分割后的列表；该函数输入三个参数，被分割的张量、分割后的维度的大小（整数或列表）、分割的维度。

4. `torch.chunk`：输入的张量在该维度的大小需要被分割的段数整除

   ```python
   t1 = torch.randn(3,4) # 随机产生四个张量
   t2 = torch.randn(3,4)
   t3 = torch.randn(3,4)
   t4 = torch.radnn(3,2) 
   
   torch.stack([t1,t2,t3], -1).shape	# 沿着最后一个维度做堆叠，返回大小为3×4×3的张量
   torch.cat([t1,t2,t3,t4], -1).shape # 沿着最后一个维度做拼接，返回大小为3×14的张量
   
   t = torch.randn(3, 6) # 随机产生一个3×6的张量
   
   t.split([1,2,3], -1) # 把张量沿着最后一个维度分割为三个张量
   t.split(3, -1) # 把张量沿着最后一个维度分割，分割大小为3，输出的张量大小均为3×3
   t.chunk(3, -1) # 把张量沿着最后一个维度分割为三个张量，大小均为3×2
   ```



#### 张量维度的扩增和压缩

可以在张量中添加任意数目的1的维度，也可以压缩这些数目为1的维度

```
t = torch.rand(3, 4) # 随机生成一个张量

t.unsqueeze(-1).shape # 扩增最后一个维度，[3,4,1]

t = torch.rand(1,3,4,1) # 随机生成一个张量，有两个维度大小为1

t.squeeze().shape # 压缩所有大小为1的维度,[3,4]
```



#### 张量的广播

两个不同维度张量之间做四则运算，且两个张量的某些维度相等

例：（3，4，5）与（3，5）相加，（3，5）—>(3，1，5)—>复制（3，4，5

```python
t1 = torch.randn(3,4,5) # 定义3×4×5的张量1
t2 = torch.randn(3,5) # 定义 3×5的张量2

t2 = t2.unsqueeze(1) # 张量2的形状变为3×1×5

t3 = t1 + t2 # 广播求和，最后结果为3×4×5的张量
```



### 4 pytorch模块简介

#### 模块类

* 模块本身是一个类nn.Module,pytorch的模型可以通过继承该类，在类的内部定义子模块的实例化，通过前向计算调用子模块，最后实现深度学习模型的搭建

  ```python
  import torch.nn as nn
  
  class Model(nn.Module):
      def __init__(self, ...): 	# 定义类的初始化函数，...是用户的传入参数
          super(Model, self).__init__()
          ... # 根据传入的参数来定义子模块
      
      def forward(self, ...): # 定义前向计算的输入参数，...一般是张量或者其他的参数
          ret = ... # 根据传入的张量和子模块计算返回张量
          return ret
  ```

  * 基于继承`nn.Module`的方法构建深度学习模块
  * 通过`__init__`方法初始化整个模型，`forward`方法对该模型进行前向计算



#### 实例化和方法调用

1. 使用`named_parameters`方法和`parameters`方法获取模型的参数

   > 通过调用`name_parameters`方法，返回的事python的一个生成器，通过访问生成器的对象得到的是该模型所有参数的名称和对应的张量值
   >
   > `parameters`方法，返回的也是一个生成器，访问生成器的结果是该模型的所有参数对应张量的值

2. 使用`train`和`eval`方法进行模型训练和测试状态的转换

   > 通过调用`train`方法会把模块（包括所有的子模块）转换到训练状态
   >
   > 调用`eval`方法会把模块（包括所有的子模块）转换到预测状态

3. 使用`named_buffers`方法和`buffers`方法获取张量的缓存

   > 通过在模块中调用`register_buffer`方法可以在模块中加入这种类型的张量
   >
   > 通过`named_buffers`可以获得缓存的名字和缓存张量的值组成的生成器
   >
   > 通过`buffers`方法可以获取缓存张量值组成的生成器

4. 使用`named_children`方法和`children`方法获取模型的子模块

   > 调用可以获取子模块名字、子模块的生成器，以及只有子模块的生成器

5. 使用`apply`方法递归地对子模块进行函数应用

   > 如果需要对pytorch所有模块应用一个函数，可以使用apply方法，通过传入一个函数或者匿名函数来递归地应用这些函数，传入的函数以模块为参数，在函数内部对这些模块进行修改

6. 改变模块参数数据类型和存储的位置

   > ```
   > lm = LinearModel(5) # 定义线性模型
   > x = torch.randn(4, 5) # 定义模型输入
   > lm(x) # 根据模型获取输入对应的输出
   > 
   > lm.named_parameters() # 获取模型参数（带名字）的生成器
   > list(lm.named_parameters()) # 转换生成器为列表
   > lm.parameters() # 获取模型参数（不带名字）的生成器
   > list(lm.parameters()) # 转换生成器为列表
   > 
   > lm.cuda() # 将模型参数移到GPU上
   > list(lm.parameters()) # 显示模型参数，可以看到已经移到了GPU上（device='cuda:0'）
   > lm.half() # 转换模型参数为半精度浮点数
   > list(lm.parameters()) # 显示模型参数，可以看到已经转换为了半精度浮点数（dtype=torch.float16）
   > ```



### 5 自动求导机制

```python
t1 = torch.randn(3, 3, requires_grad=True) # 定义一个3×3的张量

t2 = t1.pow(2).sum() # 计算张量的所有分量平方和
t2.backward() # 反向传播
t1.grad # 梯度是张量原始分量的2倍

t2 = t1.pow(2).sum() # 再次计算所有分量的平方和
t2.backward() # 再次反向传播
t1.grad # 梯度累积

t1.grad.zero_() # 单个张量清零梯度的方法
```

```python
t1 = torch.randn(3, 3, requires_grad=True) # 初始化t1张量
t2 = t1.pow(2).sum() # 根据t1张量计算t2张量

torch.autograd.grad(t2, t1) # t2张量对t1张量求导
```

可以使用torch.no_grad上下文管理器，在这个上下文管理器的作用域里进行的神经网络计算不会构建任何计算图

```python
t1 = torch.randn(3, 3, requires_grad=True) # 初始化t1张量
t2 = t1.sum()	# t2的计算构建了计算图，输出结果带有grad_fn

with torch.no_grad():
    t3 = t1.sum()# t3的计算没有构建计算图，输出结果没有grad_fn
    
t1.sum() # 保持原来的计算图
t1.sum().detach() # 和原来的计算图分离
```



### 6 损失函数和优化器

#### 损失函数

* 函数形式：调用`torch.nn.functional`库中的函数，通过传入神经网络预测值和目标值来计算损失函数
* 模块形式：torch.nn库里的模块，通过新建一个模块的实例，然后通过调用莫模块的方法来计算损失函数

```python
#解决回归问题通常使用——平方损失函数
mse = nn.MSELoss() # 初始化平方损失函数模块
t1 = torch.randn(5, requires_grad=True) # 随机生成张量t1
t2 = torch.randn(5, requires_grad=True) # 随机生成张量t2
mse(t1, t2) # 计算张量t1和t2之间的平方损失函数

#解决二分类问题——交叉熵损失函数
t1s = torch.sigmoid(t1)
t2 = torch.randint(0, 2, (5, )).float() # 随机生成0，1的整数序列，并转换为浮点数
bce(t1s, t2) # 计算二分类的交叉熵

bce_logits = nn.BCEWithLogitsLoss() # 使用交叉熵对数损失函数
bce_logits(t1, t2) # 计算二分类的交叉熵，可以发现和前面的结果一致

#多分类问题——负对数似然函数
N=10 # 定义分类数目
t1 = torch.randn(5, N, requires_grad=True) # 随机产生预测张量
t2 = torch.randint(0, N, (5, )) # 随机产生目标张量
t1s = torch.nn.functional.log_softmax(t1, -1) # 计算预测张量的LogSoftmax
nll = nn.NLLLoss() # 定义NLL损失函数
nll(t1s, t2) # 计算损失函数

ce = nn.CrossEntropyLoss() # 定义交叉熵损失函数
ce(t1, t2) # 计算损失函数，可以发现和NLL损失函数的结果一致
```



#### 优化器

```python
from sklearn.datasets import load_boston
boston = load_boston()

lm = LinearModel(13)
criterion = nn.MSELoss()
optim = torch.optim.SGD(lm.parameters(), lr=1e-6) # 定义优化器
data = torch.tensor(boston["data"], requires_grad=True, dtype=torch.float32)
target = torch.tensor(boston["target"], dtype=torch.float32)

for step in range(10000):
    predict = lm(data) # 输出模型预测结果
    loss = criterion(predict, target) # 输出损失函数
    if step and step % 1000 == 0 :
        print("Loss: {:.3f}".format(loss.item()))
    optim.zero_grad() # 清零梯度
    loss.backward() # 反向传播
    optim.step()
```

* 优化之前须执行两个步骤
  * 调用`zero_grad`方法清空所有的参数前一次反向传播的梯度，
  * 调用损失函数的`backward`方法来计算所有参数的当前反向传播的梯度



### 7 数据的输入和预处理

#### 数据载入类

```python
class torch.utils.data.DataLoader(dataset, 
									batch_size=1, 
									shuffle=False, 
									sampler=None, 
									num_workers=0, 
									collate_fn=<function default_collate>, 
									pin_memory=False,
                                    drop_last=False)
```



- **dataset** (*Dataset*) – 加载数据的数据集。
- **batch_size** (*int*, optional) – 每个batch加载多少个样本(默认: 1)。
- **shuffle** (*bool*, optional) – 设置为`True`时会在每个epoch重新打乱数据(默认: False).
- **sampler** (*Sampler*, optional) – 定义从数据集中提取样本的策略。如果指定，则忽略`shuffle`参数。
- **num_workers** (*int*, optional) – 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
- **collate_fn** (*callable*, optional) –定义如何把一批dataset的实例转换为包含迷你批次数据的张量
- **pin_memory** (*bool*, optional) –会把数据转移到和GPU内存相关联的CPU内存中，从而加快载入数据的速度
- **drop_last** (*bool*, optional) – 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)



#### 映射类型的数据集

为了使用DataLoader类，首先需要构造关于单个数据的torch.untils.data.Dataset类

```python
class Dataset(object):
    def __getitem__(self, index):
        # index: 数据缩索引（整数，范围为0到数据数目-1）
        # ...
        # 返回数据张量

    def __len__(self):
        # 返回数据的数目
        # ...
```

* `__geitem__`：对应的操作符是索引操作符[]，通过输入整数数据索引，其大小在0-N-1之间，返回具体的某一条数据记录。
* `__len__`：返回数据的总数



### 8模型的保存和加载

#### 模块和张量的序列化和反序列化

```python
torch.save(obj, f, pickle_module=pickle,pickle_protocol=2)
```

- obj – 保存对象
- f － 类文件对象 (返回文件描述符)或一个保存文件名的字符串
- pickle_module – 用于pickling元数据和对象的模块，传入序列化的库
- pickle_protocol – 指定pickle protocal 可以覆盖默认参数，把对象转换成字符串的规范

```python
torch.load(f, map_location=None, pickle_module=pickle,**pickle_load_args)
```

- f – 类文件对象 (返回文件描述符)或一个保存文件名的字符串
- map_location – 一个函数或字典规定如何remap存储位置
- pickle_module – 用于unpickling元数据和对象的模块 (必须匹配序列化文件时的pickle_module )



```python
lm = LinearModel(5) # 定义线性模型

lm.state_dict() # 获取状态字典
t = lm.state_dict() # 保存状态字典

lm = LinearModel(5) # 重新定义线性模型
lm.state_dict() # 新的状态字典，模型参数和原来的不同

lm.load_state_dict(t) # 载入原来的状态字典
lm.state_dict() # 模型参数已更新
```



#### 模块状态字典的保存和载入

```python
save_info = { # 保存的信息
    "iter_num": iter_num,  # 迭代步数 
    "optimizer": optimizer.state_dict(), # 优化器的状态字典
    "model": model.state_dict(), # 模型的状态字典
}
# 保存信息
torch.save(save_info, save_path)
# 载入信息
save_info = torch.load(save_path)
optimizer.load_state_dict(save_info["optimizer"])
model.load_state_dict(sae_info["model"])
```



## pytorch自然语言处理模块

### 1 特征提取

* 特征提取的预处理

  1. 分词（Tonkenization）
  2. 去掉停用词（Stopwords）
  3. 文本正则化（Normalization）

* 提取词频特征

  ```python
  from sklearn.feature_extraction.text import CountVectorizer
  
  vectorizer = CountVectorizer()
  corpus = [
      'This is the first document.',
      'This is the second second document.',
      'And the third one.',
      'Is this the first document?',]
  X = vectorizer.fit_transform(corpus)
  
  print(X.toarray())		#稀疏矩阵
  out：
  	[[0 1 1 1 0 0 1 0 1]
   	[0 1 0 1 0 2 1 0 1]
  	[1 0 0 0 1 0 1 1 0]
   	[0 1 1 1 0 0 1 0 1]]
  print(vectorizer.vocabulary_)	#具体对应单词
  out：
  	{'this': 8, 'is': 3, 'the': 6, 'first': 2, 'document': 1, 'second': 5, 'and': 0, 'third': 7, 'one': 4}
  ```

* TF-IDF特征

  ```python
  from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
  X = vectorizer.fit_transform(corpus)
  transformer = TfidfTransformer()
  X1 = transformer.fit_transform(X)
  X1.to_array()		#先通过CountVectorizer获取词频矩阵，后使用TfidfTransformer计算TF-IDF特征
  
  vectorizer = TfidfVectorizer()
  X2 = vectorizer.fit_transform(corpus)
  X2.toarray()		#直接使用TfidfVectorizer计算TF-IDF特征
  ```



### 2 词嵌入层

```
class torch.nn.Embedding(num_embeddings, 
						 embedding_dim, 
						 padding_idx=None, 
						 max_norm=None, 
						 norm_type=2, 
						 scale_grad_by_freq=False, 
						 sparse=False)
```

- **num_embeddings** (*[int](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/)*) - 嵌入字典的大小
- **embedding_dim** (*[int](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/)*) - 每个嵌入向量的大小
- **padding_idx** (*[int](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/), optional*) - 如果提供的话，输出遇到此下标时用零填充
- **max_norm** (*[float](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/), optional*) - 如果提供的话，会重新归一化词嵌入，使它们的范数小于提供的值
- **norm_type** (*[float](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/), optional*) - 对于max_norm选项计算p范数时的p
- **scale_grad_by_freq** (*boolean, optional*) - 如果提供的话，会根据字典中单词频率缩放梯度

```python
embedding = nn.Embedding(10, 4)
embedding.weight
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
embedding(input)

embedding = nn.Embedding(10, 4, padding_idx=0) # 定义10×4的词嵌入张量，其中索引为0的词向量为0
embedding.weight
input = torch.LongTensor([[0,2,0,5]])
embedding(input)
```



### 3 循环神经网络层

####  简单循环神经网络

```python
class torch.nn.RNN(input_size, 
				   hidden_size, 
				   num_layers,
				   nonlinearity='tanh', 
				   bias=True, 
				   batch_first=False,
				   dropout=0, 
                   bidirectional=False)
```

- input_size – 输入`x`的特征数量。
- hidden_size – 隐层的特征数量。
- num_layers – RNN的层数。
- nonlinearity – 指定非线性函数使用`tanh`还是`relu`。默认是`tanh`。
- bias – 如果是`False`，那么RNN层就不会使用偏置权重 $b_ih$和$b_hh$,默认是`True`
- batch_first – 如果`True`的话，那么输入`Tensor`的shape应该是[batch_size, time_step, feature],输出也是这样。
- dropout – 如果值非零，那么除了最后一层外，其它层的输出都会套上一个`dropout`层。
- bidirectional – 如果`True`，将会变成一个双向`RNN`，默认为`False`。



==`RNN`的输入==： **(input, h_0)** 

* input (seq_len, batch, input_size): 保存输入序列特征的`tensor`。`input`可以是被填充的变长的序列。

- h_0 (num_layers * num_directions, batch, hidden_size): 保存着初始隐状态的`tensor`。（num_directions是方向，双向为2，单向为1）

==`RNN`的输出==： **(output, h_n)**

- output (seq_len, batch, hidden_size * num_directions): 保存着`RNN`最后一层的输出特征。如果输入是被填充过的序列，那么输出也是被填充的序列。
- h_n (num_layers * num_directions, batch, hidden_size): 保存着最后一个时刻隐状态。
- **获取最后一个时间步上的输出`output[-1,:,:]`（在`seq_len`上）；获取最后一次的`hidden_state`：`h_n[-1,:,:]`；两者相等**

==`RNN`模型参数==:

- weight_ih_l[k] – 第`k`层的 `input-hidden` 权重， 可学习，形状是`(input_size x hidden_size)`。
- weight_hh_l[k] – 第`k`层的 `hidden-hidden` 权重， 可学习，形状是`(hidden_size x hidden_size)`
- bias_ih_l[k] – 第`k`层的 `input-hidden` 偏置， 可学习，形状是`(hidden_size)`
- bias_hh_l[k] – 第`k`层的 `hidden-hidden` 偏置， 可学习，形状是`(hidden_size)`



#### LSTM和GRU

```python
class torch.nn.LSTM(input_size, 
					hidden_size, 
					num_layers,
					bias=True, 
					batch_first=False, 
					dropout=0, 
					bidirectional=False)

class torch.nn.GRU(input_size, 
				   hidden_size, 
				   num_layers,
				   bias=True, 
				   batch_first=False, 
				   dropout=0, 
				   bidirectional=False)
```

==`LSTM`输入==: **（input, (h_0, c_0)）**

- input (seq_len, batch, input_size): 包含输入序列特征的`Tensor`。也可以是`packed variable` ，
- h_0 (num_layers * num_directions, batch, hidden_size):保存着`batch`中每个元素的初始化隐状态的`Tensor`
- c_0 (num_layers * num_directions, batch, hidden_size): 保存着`batch`中每个元素的初始化细胞状态的`Tensor`

==`LSTM`输出==：**（output, (h_n, c_n)）**

- output (seq_len, batch, hidden_size * num_directions): 保存`RNN`最后一层的输出的`Tensor`。 如果输入是`torch.nn.utils.rnn.PackedSequence`，那么输出也是`torch.nn.utils.rnn.PackedSequence`。
- h_n (num_layers * num_directions, batch, hidden_size): `Tensor`，保存着`RNN`最后一个时间步的隐状态。
- c_n (num_layers * num_directions, batch, hidden_size): `Tensor`，保存着`RNN`最后一个时间步的细胞状态。

==`LSTM`模型参数:==

- weight_ih_l[k] – 第`k`层可学习的`input-hidden`权重($W_{ii}|W_{if}|W_{ig}|W_{io}$)，形状为`(input_size x 4*hidden_size)`
- weight_hh_l[k] – 第`k`层可学习的`hidden-hidden`权重($W_{hi}|W_{hf}|W_{hg}|W_{ho}$)，形状为`(hidden_size x 4*hidden_size)`。
- bias_ih_l[k] – 第`k`层可学习的`input-hidden`偏置($b_{ii}|b_{if}|b_{ig}|b_{io}$)，形状为`( 4*hidden_size)`
- bias_hh_l[k] – 第`k`层可学习的`hidden-hidden`偏置($b_{hi}|b_{hf}|b_{hg}|b_{ho}$)，形状为`( 4*hidden_size)`。



### 4 自注意力机制模块

```python
torch.nn.MultiheadAttention(embed_dim, 
							num_heads, 
							dropout=0.0, 
							bias=True, 
							add_bias_kv=False, 
							add_zero_attn=False, 
							kdim=None, vdim=None)
```

- **embed_dim** – 输入模块的张量的特征维度
- **num_heads** – 代表注意力的数目.
- **dropout** – 模型最后输出的丢弃率.
- **bias** – 模型在初步生成Query、Key、Value张量的时候是否使用偏置张量.
- **add_bias_kv** – 是否在维度0给输入的Key和Value增加偏置（也就是序列长度维度上给Key和Value张量各增加偏置），默认False.
- **add_zero_attn** – 是否在迷你批次中增加一个新的批次，其中这个批次的数据为全零张量.
- **kdim** – 指定kdim的维度.
- **vdim** – 指定vdim的维度.
- **Note** – 如果kdim和vdim为None，则将它们设置为embed_dim
- key和value具有相同的特征，当输入的K、Q、V的特征维度大小不一致时，需通过kdim和vdim指定对应的K和V的维度 



```python
forward(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None)
```

- **query、key, value** –三个输入参数代表的是K、Q、V三个输入张量.
- **key_padding_mask** – 指的是对于填充字符对应的序列位置（这个张量和注意力机制的序列长度相同，是布尔型张量，大小为N×T，其中T为K、V张量的序列长度），将其注意力分数张量相同形状的张量分数置为-inf，这样对应输出的概率为0
- **need_weights** – 设置为True，则会输出最后的注意力权重张量（否则输出为None）.
- **attn_mask** – 给定的是和注意力分数张量相同形状的张量，在给定之后，新的注意力分数为原来的注意力分数张量加上attn_mask参数的输入张量得到的最终张量.



#### Transformer单层编码器和解码器模块

```python
class torch.nn.TransformerEncoderLayer(d_model,
    								   nhead, 
    								   dim_feedforward=2048, 
    								   dropout=0.1,
                                       activation='relu')
forward(src, src_mask=None, src_key_padding_mask=None)

class torch.nn.TransformerDecoderLayer(d_model,
    								   nhead, 
    								   dim_feedforward=2048, 
    								   dropout=0.1,
                                       activation='relu')
forward(tgt, memory, tgt_mask=None, memory_mask=None,
    tgt_key_padding_mask=None, memory_key_padding_mask=None)
```

- **d_model** –单层编码器模型输入的特征维度大小.
- **nhead** – 注意力的数目.
- **dim_feedforward** – 代表FF（前馈神经网络）层的两层神经网络中间层的特征数目（默认2048）.
- **dropout** – 丢弃曾的丢弃概率 (default=0.1).
- **activation** – 中间层，relu或gelu的激活功能（默认= relu）

+++++++

* **src** –编码器层的序列.
* **src_mask** – 注意力机制的掩码表示.
* **src_key_padding_mask** –源序列中的有效单词（即不包括填充单词）的掩码表示.

+++++

- **tgt** – 输入的是目标序列对应的张量.
- **memory** –源序列经过编码器输出的张量（从编码器最后一层开始的顺序）.
- **tgt_mask** – 解码器输入的子注意力机制的注意力的掩码.
- **memory_mask** –和编码器输出关联的自注意力机制的掩码.
- **tgt_key_padding_mask** –目标序列的有效单词的掩码表示.
- **memory_key_padding_mask** – 编码器输出结果的有效单词掩码表示.



#### transformer模块

```
class torch.nn.TransformerEncoder(encoder_layer, num_layers, norm=None)
forward(src, mask=None, src_key_padding_mask=None)

class torch.nn.TransformerDecoder(decoder_layer, num_layers, norm=None)
forward(tgt, memory, tgt_mask=None, memory_mask=None,
    tgt_key_padding_mask=None, memory_key_padding_mask=None)
```
- **encoder_layer** – TransformerEncoderLayer（）类的实例（必需）.
- **num_layers** – 编码器中子编码器层的数量（必需）.
- **norm** – 图层归一化组件（可选）.



```
class torch.nn.Transformer(d_model=512, 
						   nhead=8, 
						   num_encoder_layers=6,
						   num_decoder_layers=6, 
						   dim_feedforward=2048, 
						   dropout=0.1,
						   custom_encoder=None, 
						   custom_decoder=None)
forward(src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
    src_key_padding_mask=None, tgt_key_padding_mask=None,
    memory_key_padding_mask=None)
```

- **d_model** – 模型的输入序列的特征大小 (default=512).
- **nhead** – 注意力的数目 (default=8).
- **num_encoder_layers** – 编码器的子模块个数 (default=6).
- **num_decoder_layers** – 解码器的子模块个数 (default=6).
- **dim_feedforward** – FF模块的中间输出特征大小(default=2048).
- **dropout** – 丢弃概率 (default=0.1).
- **activation** – 编码器/解码器中间层，relu或gelu的激活功能。 (default=relu).
- **custom_encoder** – 自定义编码器 (default=None).
- **custom_decoder** – 自定义解码器 (default=None)