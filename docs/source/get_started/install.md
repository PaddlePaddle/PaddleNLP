# 安装

以下安装过程默认用户已安装好**paddlepaddle-gpu或paddlepaddle(版本大于或等于1.8.1)**，paddlepaddle安装方式参照[飞桨官网](https://www.paddlepaddle.org.cn/install/quick)

## pip安装

注意其中pycocotools在Windows安装较为特殊，可参考下面的Windows安装命令  

```
pip install paddlex -i https://mirror.baidu.com/pypi/simple
```

## Anaconda安装
Anaconda是一个开源的Python发行版本，其包含了conda、Python等180多个科学包及其依赖项。使用Anaconda可以通过创建多个独立的Python环境，避免用户的Python环境安装太多不同版本依赖导致冲突。  

## 代码安装

github代码会跟随开发进度不断更新

```
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX
git checkout develop
python setup.py install
```


## pycocotools安装问题

PaddleX依赖pycocotools包，如安装pycocotools失败，可参照如下方式安装pycocotools

### Windows系统
* Windows安装时可能会提示`Microsoft Visual C++ 14.0 is required`，从而导致安装出错，[点击下载VC build tools](https://go.microsoft.com/fwlink/?LinkId=691126)安装再执行如下pip命令
> 注意：安装完后，需要重新打开新的终端命令窗口

```
pip install cython
pip install git+https://gitee.com/jiangjiajun/philferriere-cocoapi.git#subdirectory=PythonAPI
```

### Linux/Mac系统
* Linux/Mac系统下，直接使用pip安装如下两个依赖即可

```
pip install cython  
pip install pycocotools
```

