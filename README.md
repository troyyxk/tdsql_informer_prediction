> write for Xuan Xie

code_template目录树

- <u>env.sh</u>: 环境配置文件,可直接运行
- experiments: 文件夹,存储配置文件
  - <u>resnet.json</u>: 配置文件举例
- lib: 文件夹,存储各种运行核心代码
  - config: 文件夹, 该文件夹下存放的是配置文件生成器和运行命令生成器
    - <u>\_\_init\_\_.py</u>: 添加路径到总路径中
    - <u>default.py</u>: 配置文件生成器
    - <u>tune_multi.py</u>: 运行命令生成器(可以不使用)
  - core: 文件夹,运行的核心文件
    - <u>function.py</u>: 训练和测试函数
    - <u>evaluate.py</u>: 测试指标函数
  - models: 文件夹,用于存储模型文件
    - <u>\_\_init\_\_.py</u>: 添加模型路径到总路径中
    - <u>resnet.py</u>: 模型文件
  - trick_dl: 文件夹,调参技巧
    - <u>earlystop.py</u>: 调参技巧,测试指标多少个epoch不下降,则lr下降10倍
    - <u>freeze.py</u>: 调参技巧,冻结部分的卷积层(可以不使用)
  - utils: 文件夹,存放相关相关需使用的函数
    - <u>modelsummary.py</u> 模型参数量打印(可以不使用)
    - <u>my_dataset.py</u> 数据集加载文件,可在其中添加处理操作
    - <u>transforms_bgzhang.py</u> 图像增强方法
    - <u>utils.py</u> 模型保存函数与log创建
- tools: 文件夹,存放训练和测试启动文件
  - <u>\_\_init\_\_.py</u>: 添加模型路径到总路径中
  - <u>train.py</u> 训练启动文件
  - <u>valid.py</u> 测试启动文件

运行方法:

先安装环境,所需要的包在`env.sh`文件中

修改一下`experiments/resnet.json`中的`TRAIN ROOT`中的路径

在`classification_template`该路径下,运行`python tool/train --cfg ./experiments/resnet.json`,就可以开始训练`valid.py`为测试文件