# python内置模块



## 1.模块介绍与导入



### 什么是模块？

在计算机程序的开发过程中，随着程序代码越写越多，在一个文件里代码就会越来越长，越来越不容易维护。

为了编写可维护的代码，我们把很多函数分组，分别放到不同的文件里，这样，每个文件包含的代码就相对较少，很多编程语言都采用这种组织代码的方式。在Python中，一个.py文件就可以称之为一个模块（Module）。



### 使用模块有什么好处？

1. 最大的好处是大大提高了代码的可维护性。其次，编写代码不必从零开始。当一个模块编写完毕，就可以被其他地方引用。我们在编写程序的时候，也经常引用其他模块，包括Python内置的模块和来自第三方的模块。
2. 使用模块还可以避免函数名和变量名冲突。每个模块有独立的命名空间，因此相同名字的函数和变量完全可以分别存在不同的模块中，所以，我们自己在编写模块时，不必考虑名字会与其他模块冲突



### 模块分类

模块分为三种：

- 内置标准模块（又称标准库）执行help(‘modules’)查看所有python自带模块列表
- 第三方开源模块，可通过pip install 模块名 联网安装
- 自定义模块



### 模块导入&调用

```python
import module_a  #导入
from module import xxfrom module.xx.xx 
import xx as rename #导入后重命令
from module.xx.xx import *  #导入一个模块下的所有方法，不建议使用module_a.xxx  #调用
```

> 注意：模块一旦被调用，即相当于执行了另外一个py文件里的代码



### 自定义模块

这个最简单， 创建一个.py文件，就可以称之为模块，就可以在另外一个程序里导入


![img](https://book.apeland.cn/media/images/2019/04/05/image.png)



### 模块查找路径

发现，自己写的模块只能在当前路径下的程序里才能导入，换一个目录再导入自己的模块就报错说找不到了， 这是为什么？

这与导入模块的查找路径有关

```python
import sys
print(sys.path)
```

输出（注意不同的电脑可能输出的不太一样）

```python
['', '/Library/Frameworks/Python.framework/Versions/3.6/lib/python36.zip', '/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6', '/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/lib-dynload', '/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages']
```

你导入一个模块时，Python解释器会按照上面列表顺序去依次到每个目录下去匹配你要导入的模块名，只要在一个目录下匹配到了该模块名，就立刻导入，不再继续往后找。

> 注意列表第一个元素为空，即代表当前目录，所以你自己定义的模块在当前目录会被优先导入。

我们自己创建的模块若想在任何地方都能调用，那就得确保你的模块文件至少在模块路径的查找列表中。

我们一般把自己写的模块放在一个带有“site-packages”字样的目录里，我们从网上下载安装的各种第三方的模块一般都放在这个目录。



## 2.第三方开源模块的安装使用



https://pypi.python.org/pypi 是python的开源模块库，截止2019年4.30日 ，已经收录了**175870**个来自全世界python开发者贡献的模块,几乎涵盖了你想用python做的任何事情。 事实上每个python开发者，只要注册一个账号就可以往这个平台上传你自己的模块，这样全世界的开发者都可以容易的下载并使用你的模块。

![img](https://book.apeland.cn/media/images/2019/04/05/pypi.png)![img](https://book.apeland.cn/media/images/2019/04/05/image_GL8rU7l.png)



### 那如何从这个平台上下载代码呢？

1.直接在上面这个页面上点download,下载后，解压并进入目录，执行以下命令完成安装

```python
编译源码    python setup.py build
安装源码    python setup.py install
```

1. 直接通过pip安装

```python
pip3 install paramiko #paramiko 是模块名
```

pip命令会自动下载模块包并完成安装。

软件一般会被自动安装你python安装目录的这个子目录里

```python
/your_python_install_path/3.6/lib/python3.6/site-packages
```

pip命令默认会连接在国外的python官方服务器下载，速度比较慢，你还可以使用国内的豆瓣源，数据会定期同步国外官网，速度快好多

```python
pip install -i 
http://pypi.douban.com/simple/ 
alex_sayhi --trusted-host 
pypi.douban.com   #alex_sayhi是模块名
```

-i 后面跟的是豆瓣源地址

—trusted-host 得加上，是通过网站https安全验证用的



### 使用

下载后，直接导入使用就可以，跟自带的模块调用方法无差，演示一个连接linux服务器并执行命令的模块

```python
import paramiko

ssh=paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.1.108', 22, 'alex', '123')

stdin, stdout, stderr = ssh.exec_command('df')
print(stdout.read())ssh.close();

执行命令 - 通过用户名和密码连接服务器
```



## 3.调用os模块



os 模块提供了很多允许你的程序与操作系统直接交互的功能

1. 得到当前工作目录，即当前Python脚本工作的目录路径: ==os.getcwd()==

2. 返回指定目录下的所有文件和目录名:==os.listdir()==

3. 函数用来删除一个文件:==os.remove()==

4. 删除多个目录：==os.removedirs（r“c：\python”）==

5. 检验给出的路径是否是一个文件：==os.path.isfile()==

6. 检验给出的路径是否是一个目录：==os.path.isdir()==

7. 判断是否是绝对路径：==os.path.isabs()==

8. 检验给出的路径是否真地存:==os.path.exists()==

9. 返回一个路径的目录名和文件名:==os.path.split()==     

   ```python
   os.path.split('/home/swaroop/byte/code/poem.txt') 
   
   结果：('/home/swaroop/byte/code', 'poem.txt') 
   ```

10. 分离扩展名：==os.path.splitext()==   

    ```python
    os.path.splitext('/usr/local/test.py')    
    
    结果：('/usr/local/test', '.py')
    ```

11. 获取路径名：==os.path.dirname()==

12. 获得绝对路径: ==os.path.abspath()==  

13. 获取文件名：==os.path.basename()==

14. 运行shell命令: ==os.system()==

15. 读取操作系统环境变量HOME的值:==os.getenv("HOME")== 

16. 返回操作系统所有的环境变量： ==os.environ== 

17. 设置系统环境变量，仅程序运行时有效：==os.environ.setdefault('HOME','/home/alex')==

18. 给出当前平台使用的行终止符:==os.linesep==    Windows使用'\r\n'，Linux and MAC使用'\n'

19. 指示你正在使用的平台：==os.name==       对于Windows，它是'nt'，而对于Linux/Unix用户，它是'posix'

20. 重命名：==os.rename（old， new）==

21. 创建多级目录：==os.makedirs（r“c：\python\test”）==

22. 创建单个目录：==os.mkdir（“test”）==

23. 获取文件属性：==os.stat（file）==

24. 修改文件权限与时间戳：==os.chmod（file）==

25. 获取文件大小：==os.path.getsize（filename）==

26. 结合目录名与文件名：==os.path.join(dir,filename)==

27. 改变工作目录到dirname: ==os.chdir(dirname)==

28. 获取当前终端的大小: ==os.get_terminal_size()==

29. 杀死进程: ==os.kill(10884,signal.SIGKILL)==




![img](https://book.apeland.cn/media/images/2019/04/10/image.png)



## 4.调用sys模块



1. ==sys.argv==           命令行参数List，第一个元素是程序本身路径
2. ==sys.exit(n)==        退出程序，正常退出时exit(0)
3. ==sys.version==        获取Python解释程序的版本信息
4. ==sys.maxint==         最大的Int值
5. ==sys.path==           返回模块的搜索路径，初始化时使用PYTHONPATH环境变量的值
6. ==sys.platform==       返回操作系统平台名称
7. ==sys.stdout.write('please:')==     #标准输出 , 引出进度条的例子， 注，在py3上不行，可以用print代替
8. ==val = sys.stdin.readline()[:-1]==          #标准输入
9. ==sys.getrecursionlimit()==                 #获取最大递归层数
10. ==sys.setrecursionlimit(1200)==        #设置最大递归层数
11. ==sys.getdefaultencoding()==            #获取解释器默认编码
12. ==sys.getfilesystemencoding==         #获取内存数据存到文件里的默认编码

 

## 5.time & datetime模块



在平常的代码中，我们常常需要与时间打交道。在Python中，与时间处理有关的模块就包括：time，datetime,calendar(很少用，不讲)，下面分别来介绍。

我们写程序时对时间的处理可以归为以下3种：

**时间的显示**，在屏幕显示、记录日志等

**时间的转换**，比如把字符串格式的日期转成Python中的日期类型

**时间的运算**，计算两个日期间的差值等



### time模块

**在Python中，通常有这几种方式来表示时间：**

1. 时间戳（timestamp）, 表示的是从1970年1月1日00:00:00开始按秒计算的偏移量。例子：1554864776.161901

2. 格式化的时间字符串，比如“2020-10-03 17:54”

3. 元组（struct_time）共九个元素。由于Python的time模块实现主要调用C库，所以各个平台可能有所不同，mac上：

   `time.struct_time(tm_year=2020, tm_mon=4, tm_mday=10, tm_hour=2, tm_min=53, tm_sec=15, tm_wday=2, tm_yday=100, tm_isdst=0)`

   ```python
   索引（Index）    属性（Attribute）    值（Values）
   0     tm_year（年）                 比如2011
   1     tm_mon（月）                  1 - 12
   2     tm_mday（日）                 1 - 31
   3     tm_hour（时）                 0 - 23
   4     tm_min（分）                  0 - 59
   5     tm_sec（秒）                  0 - 61
   6     tm_wday（weekday）            0 - 6（0表示周日）
   7     tm_yday（一年中的第几天）       1 - 366
   8     tm_isdst（是否是夏令时）        默认为-1
   ```

#### UTC时间

UTC（Coordinated Universal Time，世界协调时）亦即格林威治天文时间，世界标准时间。在中国为UTC+8，又称东8区。DST（Daylight Saving Time）即夏令时。

#### time模块的方法

1. ==time.localtime([secs])==：将一个时间戳转换为当前时区的struct_time。若secs参数未提供，则以当前时间为准。

2. ==time.gmtime([secs])==：和localtime()方法类似，gmtime()方法是将一个时间戳转换为UTC时区（0时区）的struct_time。

3. ==time.time()==：返回当前时间的时间戳。

4. ==time.mktime(t)==：将一个struct_time转化为时间戳。

5. ==time.sleep(secs)==：线程推迟指定的时间运行,单位为秒。

6. ==time.asctime([t])==：把一个表示时间的元组或者struct_time表示为这种形式：’Sun Oct 1 12:04:38 2019’。如果没有参数，将会将time.localtime()作为参数传入。

7. ==time.ctime([secs])==：把一个时间戳（按秒计算的浮点数）转化为time.asctime()的形式。如果参数未给或者为None的时候，将会默认time.time()为参数。它的作用相当于time.asctime(time.localtime(secs))。

8. ==time.strftime(format[, t])==：把一个代表时间的元组或者struct_time（如由time.localtime()和time.gmtime()返回）转化为格式化的时间字符串。如果t未指定，将传入time.localtime()。

  - 举例：time.strftime(“%Y-%m-%d %X”, time.localtime())
  -  输出’2017-10-01 12:14:23’

9. ==time.strptime(string[, format])==：把一个格式化时间字符串转化为struct_time。实际上它和strftime()是逆操作。

  - 举例：time.strptime(‘2017-10-3 17:54’,   ”%Y-%m-%d %H:%M”) 
  - 输出 time.struct_time(tm_year=2017, tm_mon=10, tm_mday=3, tm_hour=17, tm_min=54, tm_sec=0, tm_wday=1, tm_yday=276, tm_isdst=-1)

10. 字符串转时间格式对应表

  |      | Meaning                                                      | Notes                     |
  | ---- | ------------------------------------------------------------ | ------------------------- |
  | `%a` | Locale’s abbreviated weekday name.                           |                           |
  | `%A` | Locale’s full weekday name.                                  |                           |
  | `%b` | Locale’s abbreviated month name.                             |                           |
  | `%B` | Locale’s full month name.                                    |                           |
  | `%c` | Locale’s appropriate date and time representation.           |                           |
  | `%d` | Day of the month as a decimal number [01,31].                |                           |
  | `%H` | Hour (24-hour clock) as a decimal number [00,23].            |                           |
  | `%I` | Hour (12-hour clock) as a decimal number [01,12].            |                           |
  | `%j` | Day of the year as a decimal number [001,366].               |                           |
  | `%m` | Month as a decimal number [01,12].                           |                           |
  | `%M` | Minute as a decimal number [00,59].                          |                           |
  | `%p` | Locale’s equivalent of either AM or PM.                      | (1)                       |
  | `%S` | Second as a decimal number [00,61].                          | (2)                       |
  | `%U` | Week number of the year (Sunday as the first day of the week) as a decimal number [00,53]. All days in a new year preceding the first Sunday are considered to be in week 0. | (3)                       |
  | `%w` | Weekday as a decimal number [0(Sunday),6].                   |                           |
  | `%W` | Week number of the year (Monday as the first day of the week) as a decimal number [00,53]. All days in a new year preceding the first Monday are considered to be in week 0. | (3)                       |
  | `%x` | Locale’s appropriate date representation.                    |                           |
  | `%X` | Locale’s appropriate time representation.                    |                           |
  | `%y` | Year without century as a decimal number [00,99].            |                           |
  | `%Y` | Year with century as a decimal number.                       |                           |
  | `%z` | Time zone offset indicating a positive or negative time difference from UTC/GMT of the form +HHMM or -HHMM, where H represents decimal hour digits and M represents decimal minute digits [-23:59, +23:59]. |                           |
  | `%Z` | Time zone name (no characters if no time zone exists).       |                           |
  |      | `%%`                                                         | A literal `‘%’`character. |

最后为了容易记住转换关系，看下图

![img](https://book.apeland.cn/media/images/2019/04/10/time-convert.png)



### datetime模块

相比于time模块，datetime模块的接口则更直观、更容易调用

**datetime模块定义了下面这几个类：**

1. ==datetime.date==：表示日期的类。常用的属性有year, month, day；
2. ==datetime.time==：表示时间的类。常用的属性有hour, minute, second, microsecond；
3. ==datetime.datetime==：表示日期时间。
4. ==datetime.timedelta==：表示时间间隔，即两个时间点之间的长度。
5. ==datetime.tzinfo==：与时区有关的相关信息。（这里不详细充分讨论该类，感兴趣的童鞋可以参考python手册）

**我们需要记住的方法仅以下几个：**

1. d=datetime.datetime.now() 返回当前的datetime日期类型

```python
d.timestamp(),d.today(), 
d.year,d.timetuple()等方法可以调用
```

2. datetime.date.fromtimestamp(322222) 把一个时间戳转为datetime日期类型

3. 时间运算

```python
>>> datetime.datetime.now()

datetime.datetime(2017, 10, 1, 12, 53, 11, 821218)

>>> datetime.datetime.now() + datetime.timedelta(4)		 #当前时间 +4天

datetime.datetime(2017, 10, 5, 12, 53, 35, 276589)

>>> datetime.datetime.now() + datetime.timedelta(hours=4) 	#当前时间+4小时

datetime.datetime(2017, 10, 1, 16, 53, 42, 876275)
```

4. 时间替换

```python
>>> d.replace(year=2999,month=11,day=30)

datetime.date(2999, 11, 30)
```



## 6.random随机模块



程序中有很多地方需要用到随机字符，

比如登录网站的随机验证码，通过random模块可以很容易生成随机字符串



1. ==random.randrange(1,10)== 		#返回1-10之间的一个随机数，不包括10

2. ==random.randint(1,10)==        #返回1-10之间的一个随机数，包括10

3. ==random.randrange(0, 100, 2)==       #随机选取0到100间的偶数

4.  ==random.random()==         #返回一个随机浮点数

5.  ==random.choice('abce3#$@1')==          #返回一个给定数据集合中的随机字符'#'

6. ==random.sample('abcdefghij',3)==         #从多个字符中选取特定数量的字符['a', 'd', 'b']

   ```python
   #生成随机字符串
   >>> import string 
   >>> ''.join(random.sample(string.ascii_lowercase + string.digits, 6)) 	
   										#小写英文字母和小写数字
   '4fvda1'
   
   #洗牌
   >>> a[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
   >>> random.shuffle(a)
   >>> a
   
   [3, 0, 7, 2, 1, 6, 5, 8, 9, 4]
   ```



## 7.序列化pickle&json模块



#### 什么叫序列化？

序列化是指把内存里的数据类型转变成字符串，以使其能存储到硬盘或通过网络传输到远程，因为硬盘或网络传输时只能接受bytes

#### 为什么要序列化？

你打游戏过程中，打累了，停下来，关掉游戏、想过2天再玩，2天之后，游戏又从你上次停止的地方继续运行，你上次游戏的进度肯定保存在硬盘上了，是以何种形式呢？游戏过程中产生的很多临时数据是不规律的，可能在你关掉游戏时正好有10个列表，3个嵌套字典的数据集合在内存里，需要存下来？你如何存？把列表变成文件里的多行多列形式？那嵌套字典呢？根本没法存。所以，若是有种办法可以直接把内存数据存到硬盘上，下次程序再启动，再从硬盘上读回来，还是原来的格式的话，那是极好的。

用于序列化的两个模块

- json，用于字符串 和 python数据类型间进行转换
- pickle，用于python特有的类型 和 python的数据类型间进行转换

pickle模块提供了四个功能：dumps、dump、loads、load

```python
import pickle
data = {'k1':123,'k2':'Hello'}

# pickle.dumps 将数据通过特殊的形式转换位只有python语言认识的字符串
p_str = pickle.dumps(data) 		 # 注意dumps会把数据变成bytes格式
print(p_str)
b'\x80\x03}q\x00(X\x02\x00\x00\x00k2q\x01X\x05\x00\x00\x00Helloq\x02X\x02\x00\x00\x00k1q\x03K{u.'

# pickle.dump 将数据通过特殊的形式转换位只有python语言认识的字符串，并写入文件
with open('result.pk',"wb") as fp:    
	pickle.dump(data,fp)
	
# pickle.load  从文件里加载
f = open("result.pk","rb")
d = pickle.load(f)
print(d)
{'k1': 123, 'k2': 'Hello'}
```



Json模块也提供了四个功能：dumps、dump、loads、load，用法跟pickle一致

```python
import json

# json.dumps 将数据通过特殊的形式转换位所有程序语言都认识的字符串
j_str = json.dumps(data)       # 注意json dumps生成的是字符串，不是bytes
print(j_str)

#dump入文件 
with open('result.json','w') as fp:    
    json.dump(data,fp)   

#从文件里load
with open("result.json") as f:    
    d = json.load(f)    
    print(d)
```



#### json vs pickle:

- **JSON:**
  - 优点：跨语言(不同语言间的数据传递可用json交接)、体积小
  - 缺点：只能支持int\str\list\tuple\dict

- **Pickle:**
  - 优点：专为python设计，支持python所有的数据类型
  - 缺点：只能在python中使用，存储数据占空间大



## 8.hashlib 加密



### 加密算法介绍

### HASH

Hash，一般翻译做“散列”，也有直接音译为”哈希”的，就是把任意长度的输入（又叫做预映射，pre-image），通过散列算法，变换成固定长度的输出，该输出就是散列值。这种转换是一种压缩映射，也就是，散列值的空间通常远小于输入的空间，不同的输入可能会散列成相同的输出，而不可能从散列值来唯一的确定输入值。

简单的说就是一种将任意长度的消息压缩到某一固定长度的消息摘要的函数。

HASH主要用于信息安全领域中加密算法，他把一些不同长度的信息转化成杂乱的128位的编码里,叫做HASH值.也可以说，hash就是找到一种数据内容和数据存放地址之间的映射关系

### MD5

**什么是MD5算法**

MD5讯息摘要演算法（英语：MD5 Message-Digest Algorithm），一种被广泛使用的密码杂凑函数，可以产生出一个128位的散列值（hash value），用于确保信息传输完整一致。MD5的前身有MD2、MD3和MD4。

**MD5功能**

输入任意长度的信息，经过处理，输出为128位的信息（数字指纹）；

不同的输入得到的不同的结果（唯一性）；

**MD5算法的特点**

1. 压缩性：任意长度的数据，算出的MD5值的长度都是固定的
2. 容易计算：从原数据计算出MD5值很容易
3. 抗修改性：对原数据进行任何改动，修改一个字节生成的MD5值区别也会很大
4. 强抗碰撞：已知原数据和MD5，想找到一个具有相同MD5值的数据（即伪造数据）是非常困难的。

**MD5算法是否可逆？**

MD5不可逆的原因是其是一种散列函数，使用的是hash算法，在计算过程中原文的部分信息是丢失了的。

**MD5用途**

1. 防止被篡改：
   - 比如发送一个电子文档，发送前，我先得到MD5的输出结果a。然后在对方收到电子文档后，对方也得到一个MD5的输出结果b。如果a与b一样就代表中途未被篡改。
   - 比如我提供文件下载，为了防止不法分子在安装程序中添加木马，我可以在网站上公布由安装文件得到的MD5输出结果。
   - SVN在检测文件是否在CheckOut后被修改过，也是用到了MD5.
2. 防止直接看到明文：
   - 现在很多网站在数据库存储用户的密码的时候都是存储用户密码的MD5值。这样就算不法分子得到数据库的用户密码的MD5值，也无法知道用户的密码。（比如在UNIX系统中用户的密码就是以MD5（或其它类似的算法）经加密后存储在文件系统中。当用户登录的时候，系统把用户输入的密码计算成MD5值，然后再去和保存在文件系统中的MD5值进行比较，进而确定输入的密码是否正确。通过这样的步骤，系统在并不知道用户密码的明码的情况下就可以确定用户登录系统的合法性。这不但可以避免用户的密码被具有系统管理员权限的用户知道，而且还在一定程度上增加了密码被破解的难度。）
3. 防止抵赖（数字签名）：
   - 这需要一个第三方认证机构。例如A写了一个文件，认证机构对此文件用MD5算法产生摘要信息并做好记录。若以后A说这文件不是他写的，权威机构只需对此文件重新产生摘要信息，然后跟记录在册的摘要信息进行比对，相同的话，就证明是A写的了。这就是所谓的“数字签名”。

### SHA-1

安全哈希算法（Secure Hash Algorithm）主要适用于数字签名标准（Digital Signature Standard DSS）里面定义的数字签名算法（Digital Signature Algorithm DSA）。对于长度小于2^64位的消息，SHA1会产生一个160位的消息摘要。当接收到消息的时候，这个消息摘要可以用来验证数据的完整性。

SHA是美国国家安全局设计的，由美国国家标准和技术研究院发布的一系列密码散列函数。

由于MD5和SHA-1于2005年被山东大学的教授王小云破解了，科学家们又推出了SHA224, SHA256, SHA384, SHA512，当然位数越长，破解难度越大，但同时生成加密的消息摘要所耗时间也更长。**目前最流行的是加密算法是SHA-256 .**

### MD5与SHA-1的比较

由于MD5与SHA-1均是从MD4发展而来，它们的结构和强度等特性有很多相似之处，SHA-1与MD5的最大区别在于其摘要比MD5摘要长32 比特。对于强行攻击，产生任何一个报文使之摘要等于给定报文摘要的难度：MD5是2128数量级的操作，SHA-1是2160数量级的操作。产生具有相同摘要的两个报文的难度：MD5是264是数量级的操作，SHA-1 是280数量级的操作。因而,SHA-1对强行攻击的强度更大。但由于SHA-1的循环步骤比MD5多80:64且要处理的缓存大160比特:128比特，SHA-1的运行速度比MD5慢。

### Python的 提供的相关模块

用于加密相关的操作，3.x里用hashlib代替了md5模块和sha模块，主要提供 SHA1, SHA224, SHA256, SHA384, SHA512 ，MD5 算法

```python
import hashlib

# md5
m = hashlib.md5()
m.update(b"Hello")
m.update(b"It's me")
print(m.digest())  # 返回2进制格式的hash值
m.update(b"It's been a long time since last time we ...")
print(m.hexdigest()) # 返回16进制格式的hash值

# sha1
s1 = hashlib.sha1()
s1.update("小猿圈".encode("utf-8"))
s1.hexdigest()

# sha256
s256 = hashlib.sha256()
s256.update("小猿圈".encode("utf-8"))
s256.hexdigest()

# sha512
s512 = hashlib.sha256()
s512.update("小猿圈".encode("utf-8"))
s512.hexdigest()
```



## 9.文件copy模块shutil



### shutil 模块

高级的文件、文件夹、压缩包 处理模块

==**shutil.copyfileobj(fsrc, fdst[, length])**==

将文件内容拷贝到另一个文件中

```python
import shutil
shutil.copyfileobj(open('old.xml','r'), open('new.xml', 'w'))
```

==**shutil.copyfile(src, dst)**==

拷贝文件

```python
shutil.copyfile('f1.log', 'f2.log') #目标文件无需存在
```

==**shutil.copymode(src, dst)**==

仅拷贝权限。内容、组、用户均不变

```python
shutil.copymode('f1.log', 'f2.log') #目标文件必须存在
```

==**shutil.copystat(src, dst)**==

仅拷贝状态的信息，包括：mode bits, atime, mtime, flags

```python
shutil.copystat('f1.log', 'f2.log') #目标文件必须存在
```

**==shutil.copy(src, dst)==**

拷贝文件和权限

```python
import shutilshutil.copy('f1.log', 'f2.log')
```

==**shutil.copy2(src, dst)**==

拷贝文件和状态信息

```python
import shutilshutil.copy2('f1.log', 'f2.log')
```

==**shutil.ignore_patterns(\*patterns)**==

==**shutil.copytree(src, dst, symlinks=False, ignore=None)**==

递归的去拷贝文件夹

```python
import shutilshutil.copytree('folder1', 'folder2', 				    	
								ignore=shutil.ignore_patterns('*.pyc', 'tmp*'))
#目标目录不能存在，注意对folder2目录父级目录要有可写权限，ignore的意思是排除
```

==**shutil.rmtree(path[, ignore_errors[, onerror]])**==

递归的去删除文件

```python
import shutilshutil.rmtree('folder1')
```

**==shutil.move(src, dst)==**

递归的去移动文件，它类似mv命令，其实就是重命名。

```python
import shutilshutil.move('folder1', 'folder3')
```

==**shutil.make_archive(base_name, format,…)**==

创建压缩包并返回文件路径，例如：zip、tar

可选参数如下：

- base_name： 压缩包的文件名，也可以是压缩包的路径。只是文件名时，则保存至当前目录，否则保存至指定路径，

如 data_bak =>保存至当前路径

如：/tmp/data_bak =>保存至/tmp/

- format： 压缩包种类，“zip”, “tar”, “bztar”，“gztar”
- root_dir： 要压缩的文件夹路径（默认当前目录）
- owner： 用户，默认当前用户
- group： 组，默认当前组
- logger： 用于记录日志，通常是logging.Logger对象

```python
#将 /data 下的文件打包放置当前程序目录
import shutil
ret = shutil.make_archive("data_bak", 'gztar', root_dir='/data')

#将 /data下的文件打包放置 /tmp/目录
import shutil
ret = shutil.make_archive("/tmp/data_bak", 'gztar', root_dir='/data')
```

shutil 对压缩包的处理是调用 ZipFile 和 TarFile 两个模块来进行的，详细：

zipfile压缩&解压缩

```python
import zipfile
# 压缩
z = zipfile.ZipFile('laxi.zip', 'w')
z.write('a.log')
z.write('data.data')
z.close()

# 解压
z = zipfile.ZipFile('laxi.zip', 'r')
z.extractall(path='.')
z.close()
```

tarfile压缩&解压缩

```python
import tarfile
# 压缩
>>> t=tarfile.open('/tmp/egon.tar','w')
>>> t.add('/test1/a.py',arcname='a.bak')
>>> t.add('/test1/b.py',arcname='b.bak')
>>> t.close()

# 解压
>>> t=tarfile.open('/tmp/egon.tar','r')
>>> t.extractall('/egon')
>>> t.close()
```



## 10.正则表达式re模块



### 引子

请从以下文件里取出所有的手机号

```markdown
姓名        地区    身高    体重    电话
况咏蜜     北京    171    48    13651054608
王心颜     上海    169    46    13813234424
马纤羽     深圳    173    50    13744234523
乔亦菲     广州    172    52    15823423525
罗梦竹     北京    175    49    18623423421
刘诺涵     北京    170    48    18623423765
岳妮妮     深圳    177    54    18835324553
贺婉萱     深圳    174    52    18933434452
叶梓萱     上海    171    49    18042432324
杜姗姗     北京    167    49    13324523342
```

你能想到的办法是什么？

必然是下面这种吧？

```python
f = open("兼职白领学生空姐模特护士联系方式.txt",'r',encoding="gbk")
phones = []
for line in f:    
	name,city,height,weight,phone = line.split()    
	if phone.startswith('1') and len(phone) == 11:        			
		phones.append(phone)
print(phones)
```



有没有更简单的方式？

手机号是有规则的，都是数字且是11位，再严格点，就都是1开头，如果能把这样的规则写成代码，直接拿规则代码匹配文件内容不就行了？

![img](https://book.apeland.cn/media/images/2019/04/10/image_jNixbI5.png)

这么nb的玩法是什么？它的名字叫正则表达式！

### re模块

正则表达式就是字符串的匹配规则，在多数编程语言里都有相应的支持，python里对应的模块是re

#### 常用的表达式规则

```python
'.'     默认匹配除\n之外的任意一个字符，若指定flag DOTALL,则匹配任意字符，包括换行

'^'     匹配字符开头，若指定flags MULTILINE,这种也可以匹配上(r"^a","\nabc\neee",flags=re.MULTILINE)

'$'     匹配字符结尾， 若指定flags MULTILINE ,re.search('foo.$','foo1\nfoo2\n',re.MULTILINE).group() 会匹配到foo1

'*'     匹配*号前的字符0次或多次， re.search('a*','aaaabac')  结果'aaaa'

'+'     匹配前一个字符1次或多次，re.findall("ab+","ab+cd+abb+bba") 结果['ab', 'abb']

'?'     匹配前一个字符1次或0次 ,re.search('b?','alex').group() 匹配b 0次

'{m}'   匹配前一个字符m次 ,re.search('b{3}','alexbbbs').group()  匹配到'bbb'

'{n,m}' 匹配前一个字符n到m次，re.findall("ab{1,3}","abb abc abbcbbb") 结果'abb', 'ab', 'abb']

'|'     匹配|左或|右的字符，re.search("abc|ABC","ABCBabcCD").group() 结果'ABC'

'(...)' 分组匹配， re.search("(abc){2}a(123|45)", "abcabca456c").group() 结果为'abcabca45'

'\A'    只从字符开头匹配，re.search("\Aabc","alexabc") 是匹配不到的，相当于re.match('abc',"alexabc") 或^

'\Z'    匹配字符结尾，同$ 
'\d'    匹配数字0-9
'\D'    匹配非数字
'\w'    匹配[A-Za-z0-9]
'\W'    匹配非[A-Za-z0-9]
's'     匹配空白字符、\t、\n、\r , re.search("\s+","ab\tc1\n3").group() 结果 '\t'

'(?P...)' 分组匹配 re.search("(?P[0-9]{4})(?P[0-9]{2})(?P[0-9]{4}
```

#### re的匹配语法有以下几种

- ==re.match== 从头开始匹配
- ==re.search== 匹配包含
- ==re.findall== 把所有匹配到的字符放到以列表中的元素返回
- ==re.split== 以匹配到的字符当做列表分隔符
- ==re.sub== 匹配字符并替换
- ==re.fullmatch== 全部匹配

#### ==re.compile(pattern, flags=0)==

将正则表达式模式编译成正则表达式对象，可使用其match()，search()和其他方法（如下所述）进行匹配。

序列

```python
prog = re.compile(pattern)
result = prog.match(string)
```

相当于

```python
result = re.match(pattern, string)
```

但是，如果在单个程序中多次使用该表达式，则使用re.compile()并保存生成的正则表达式对象以供重用会更有效。

#### ==re.match(pattern, string, flags=0)==

从起始位置开始根据模型去字符串中匹配指定内容，匹配单个

- pattern 正则表达式
- string 要匹配的字符串
- flags 标志位，用于控制正则表达式的匹配方式

```python
import re
obj = re.match('\d+', '123uuasf') #如果能匹配到就返回一个可调用的对象，否则返回None
if obj:     
    print obj.group()
```

#### Flags标志符

- ==re.I(re.IGNORECASE)==: 忽略大小写（括号内是完整写法，下同）
- ==re.M(MULTILINE)==: 多行模式，改变’^’和’$’的行为
- ==re.S(DOTALL)==: 改变’.’的行为,make the ‘.’ special character match any character at all, including a newline; without this flag, ‘.’ will match anything except a newline.
- ==re.X(re.VERBOSE)== 可以给你的表达式写注释，使其更可读，下面这2个意思一样

```python
a = re.compile(r"""\d + # the integral part                
					\. # the decimal point                
					\d * # some fractional digits""",                 
					re.X)
					
b = re.compile(r"\d+\.\d*")
```

#### ==re.search(pattern, string, flags=0)==

根据模型去字符串中匹配指定内容，匹配单个

```python
import re
obj = re.search('\d+', 'u123uu888asf')
if obj:    
    print obj.group()
```

#### ==re.findall(pattern, string, flags=0)==

match and search均用于匹配单值，即：只能匹配字符串中的一个，如果想要匹配到字符串中所有符合条件的元素，则需要使用 findall。

```python
import re
obj = re.findall('\d+', 'fa123uu888asf')
print obj
```

#### ==**re.sub(pattern, repl, string, count=0, flags=0)**==

用于替换匹配的字符串,比str.replace功能更加强大

```python
>>>re.sub('[a-z]+','sb','武配齐是abc123',)
>>> re.sub('\d+','|', 'alex22wupeiqi33oldboy55',count=2)
'alex|wupeiqi|oldboy55'
```

#### ==re.split(pattern, string, maxsplit=0, flags=0)==

用匹配到的值做为分割点，把值分割成列表

```python
>>>s='9-2*5/3+7/3*99/4*2998+10*568/14'
>>>re.split('[\*\-\/\+]',s)['9', '2', '5', '3', '7', '3', '99', '4', '2998', '10', 					'568', '14']
>>> re.split('[\*\-\/\+]',s,3)
['9', '2', '5', '3+7/3*99/4*2998+10*568/14']
```

#### ==**re.fullmatch(pattern, string, flags=0)**==

整个字符串匹配成功就返回re object, 否则返回None

```python
re.fullmatch('\w+@\w+\.(com|cn|edu)',"alex@oldboyedu.cn")
```



## 11.软件开发目录设计规范



### 为什么要设计好目录结构?

“设计项目目录结构”，就和”代码编码风格”一样，属于个人风格问题。对于这种风格上的规范，一直都存在两种态度:

1. 一类同学认为，这种个人风格问题”无关紧要”。理由是能让程序work就好，风格问题根本不是问题。
2. 另一类同学认为，规范化能更好的控制程序结构，让程序具有更高的可读性。

我是比较偏向于后者的，因为我是前一类同学思想行为下的直接受害者。我曾经维护过一个非常不好读的项目，其实现的逻辑并不复杂，但是却耗费了我非常长的时间去理解它想表达的意思。从此我个人对于提高项目可读性、可维护性的要求就很高了。”项目目录结构”其实也是属于”可读性和可维护性”的范畴，我们设计一个层次清晰的目录结构，就是为了达到以下两点:

1. **可读性高**: 不熟悉这个项目的代码的人，一眼就能看懂目录结构，知道程序启动脚本是哪个，测试目录在哪儿，配置文件在哪儿等等。从而非常快速的了解这个项目。
2. **可维护性高**: 定义好组织规则后，维护者就能很明确地知道，新增的哪个文件和代码应该放在什么目录之下。这个好处是，随着时间的推移，代码/配置的规模增加，项目结构不会混乱，仍然能够组织良好。

所以，我认为，保持一个层次清晰的目录结构是有必要的。更何况组织一个良好的工程目录，其实是一件很简单的事儿。

### 目录组织方式

关于如何组织一个较好的Python工程目录结构，已经有一些得到了共识的目录结构。在Stackoverflow的[这个问题](http://stackoverflow.com/questions/193161/what-is-the-best-project-structure-for-a-python-application)上，能看到大家对Python目录结构的讨论。

这里面说的已经很好了，我也不打算重新造轮子列举各种不同的方式，这里面我说一下我的理解和体会。

假设你的项目名为foo, 我比较建议的最方便快捷目录结构这样就足够了:

```
Foo/
|-- bin/
|   |-- foo
|
|-- foo/
|   |-- tests/
|   |   |-- __init__.py
|   |   |-- test_main.py
|   |
|   |-- __init__.py
|   |-- main.py
|
|-- docs/
|   |-- conf.py
|   |-- abc.rst
|
|-- setup.py
|-- requirements.txt
|-- README
```

简要解释一下:

- bin/: 存放项目的一些可执行文件，当然你可以起名script/之类的也行。
- foo/: 存放项目的所有源代码。(1) 源代码中的所有模块、包都应该放在此目录。不要置于顶层目录。(2) 其子目录tests/存放单元测试代码； (3) 程序的入口最好命名为main.py。
- docs/: 存放一些文档。
- setup.py: 安装、部署、打包的脚本。
- requirements.txt: 存放软件依赖的外部Python包列表。
- README: 项目说明文件。

除此之外，有一些方案给出了更加多的内容。比如`LICENSE.txt`,`ChangeLog.txt`文件等，我没有列在这里，因为这些东西主要是项目开源的时候需要用到。如果你想写一个开源软件，目录该如何组织，可以参考[这篇文章](http://www.jeffknupp.com/blog/2013/08/16/open-sourcing-a-python-project-the-right-way/)。

下面，再简单讲一下我对这些目录的理解和个人要求吧。

### 关于README的内容

**这个我觉得是每个项目都应该有的一个文件**，目的是能简要描述该项目的信息，让读者快速了解这个项目。

它需要说明以下几个事项:

1. 软件定位，软件的基本功能。
2. 运行代码的方法: 安装环境、启动命令等。
3. 简要的使用说明。
4. 代码目录结构说明，更详细点可以说明软件的基本原理。
5. 常见问题说明。

### 关于requirements.txt和setup.py

#### setup.py

> 一般来说，用`setup.py`来管理代码的打包、安装、部署问题。业界标准的写法是用Python流行的打包工具[setuptools](https://pythonhosted.org/setuptools/setuptools.html#developer-s-guide)来管理这些事情。这种方式普遍应用于开源项目中。不过这里的核心思想不是用标准化的工具来解决这些问题，而是说，**一个项目一定要有一个安装部署工具**，能快速便捷的在一台新机器上将环境装好、代码部署好和将程序运行起来。

这个我是踩过坑的。

我刚开始接触Python写项目的时候，安装环境、部署代码、运行程序这个过程全是手动完成，遇到过以下问题:

1. 安装环境时经常忘了最近又添加了一个新的Python包，结果一到线上运行，程序就出错了。
2. Python包的版本依赖问题，有时候我们程序中使用的是一个版本的Python包，但是官方的已经是最新的包了，通过手动安装就可能装错了。
3. 如果依赖的包很多的话，一个一个安装这些依赖是很费时的事情。
4. 新同学开始写项目的时候，将程序跑起来非常麻烦，因为可能经常忘了要怎么安装各种依赖。

`setup.py`可以将这些事情自动化起来，提高效率、减少出错的概率。”复杂的东西自动化，能自动化的东西一定要自动化。”是一个非常好的习惯。

setuptools的[文档](https://pythonhosted.org/setuptools/setuptools.html#developer-s-guide)比较庞大，刚接触的话，可能不太好找到切入点。学习技术的方式就是看他人是怎么用的，可以参考一下Python的一个Web框架，flask是如何写的: [setup.py](https://github.com/mitsuhiko/flask/blob/master/setup.py)

当然，简单点自己写个安装脚本（`deploy.sh`）替代`setup.py`也未尝不可。

#### requirements.txt

这个文件存在的目的是:

1. 方便开发者维护软件的包依赖。将开发过程中新增的包添加进这个列表中，避免在 `setup.py` 安装依赖时漏掉软件包。
2. 方便读者明确项目使用了哪些Python包。

这个文件的格式是每一行包含一个包依赖的说明，通常是`flask>=0.10`这种格式，要求是这个格式能被`pip`识别，这样就可以简单的通过 `pip install -r requirements.txt`来把所有Python包依赖都装好了。具体格式说明： [点这里](https://pip.readthedocs.org/en/1.1/requirements.html)。



### 关于配置文件的使用方法

注意，在上面的目录结构中，没有将`conf.py`放在源码目录下，而是放在`docs/`目录下。

很多项目对配置文件的使用做法是:

1. 配置文件写在一个或多个python文件中，比如此处的conf.py。
2. 项目中哪个模块用到这个配置文件就直接通过`import conf`这种形式来在代码中使用配置。

这种做法我不太赞同:

1. 这让单元测试变得困难（因为模块内部依赖了外部配置）
2. 另一方面配置文件作为用户控制程序的接口，应当可以由用户自由指定该文件的路径。
3. 程序组件可复用性太差，因为这种贯穿所有模块的代码硬编码方式，使得大部分模块都依赖`conf.py`这个文件。

所以，我认为配置的使用，更好的方式是，

1. 模块的配置都是可以灵活配置的，不受外部配置文件的影响。
2. 程序的配置也是可以灵活控制的。

能够佐证这个思想的是，用过nginx和mysql的同学都知道，nginx、mysql这些程序都可以自由的指定用户配置。

所以，不应当在代码中直接`import conf`来使用配置文件。上面目录结构中的`conf.py`，是给出的一个配置样例，不是在写死在程序中直接引用的配置文件。可以通过给`main.py`启动参数指定配置路径的方式来让程序读取配置内容。当然，这里的`conf.py`你可以换个类似的名字，比如`settings.py`。或者你也可以使用其他格式的内容来编写配置文件，比如`settings.yaml`之类的。



## 12.包&跨模块代码调用

当你的模块文件越来越多，就需要对模块文件进行划分，比如把负责跟数据库交互的都放一个文件夹，把与页面交互相关的放一个文件夹，

```
my_proj/
├── apeland_web  #代码目录
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   └── views.py
├── manage.py
└── my_proj    #配置文件目录
	├── __init__.py
    ├── settings.py
    ├── urls.py
    └── wsgi.py
```

像上面这样，**一个文件夹管理多个模块文件，这个文件夹就被称为包**

一个包就是一个文件夹，但该文件夹下必须存在 **init**.py 文件, 该文件的内容可以为空， **int**.py用于标识当前文件夹是一个包。

这个**init**.py的文件主要是用来对包进行一些初始化的，当当前这个package被别的程序调用时，**init**.py文件会先执行，一般为空， 一些你希望只要package被调用就立刻执行的代码可以放在**init**.py里，一会后面会演示。



#### 跨模块导入

目录结构如下

```
my_proj
├── apeland_web
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   └── views.py
├── manage.py
└── my_proj
	├── settings.py
    ├── urls.py
    └── wsgi.py
```

根据上面的结构，如何实现在apeland*web`/views.py`里导入my*`proj/settings.py`模块？

直接导入的话，会报错，说找到不模块

![img](https://book.apeland.cn/media/images/2019/04/11/image.png)

是因为路径找不到，my_proj/settings.py 相当于是apeland_web/views.py的父亲(apeland_web)的兄弟(my_proj)的儿子(settings.py)，settings.py算是views.py的表弟啦，在views.py里只能导入同级别兄弟模块代码，或者子级别包里的模块，根本不知道表弟表哥的存在。这可怎么办呢？

答案是**添加环境变量，把父亲级的路径添加到sys.path中，就可以了，这样导入 就相当于从父亲级开始找模块了。**

*apeland_web/views.py中添加环境变量*

```
import sys ,os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
							#__file__的是打印当前被执行的模块.py文件相对路径，注意是相对路径
print(BASE_DIR) 	
			# 输出是/Users/alex/PycharmProjects/apeland_py_learn/day4_常用模块/my_proj 

sys.path.append(BASE_DIR)

from  my_proj import settings
print(settings.DATABASES)
```

### 官方推荐的跨目录导入方法

虽然通过添加环境变量的方式可以实现跨模块导入，但是官方不推荐这么干，因为这样就需要在每个目录下的每个程序里都写一遍添加环境变量的代码。

官方推荐的玩法是，在项目里创建个入口程序，整个程序调用的开始应该是从入口程序发起，这个入口程序一般放在项目的顶级目录

这样做的好处是，项目中的二级目录 apeland_web/views.py中再调用他表亲my_proj/settings.py时就不用再添加环境变量了。

原因是由于manage.py在顶层，manage.py启动时项目的环境变量路径就会自动变成….xxx/my_proj/这一级别

![img](https://book.apeland.cn/media/images/2019/04/11/image_FMkmC3b.png)

