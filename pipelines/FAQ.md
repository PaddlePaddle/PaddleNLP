## FAQ

#### pip安装htbuilder包报错，`UnicodeDecodeError: 'gbk' codec can't decode byte....`

windows的默认字符gbk导致的，可以使用源码进行安装，源码已经进行了修复。

```
git clone https://github.com/tvst/htbuilder.git
cd htbuilder/
python setup install
```

#### 语义检索系统可以跑通，但终端输出字符是乱码怎么解决？

+ 通过如下命令设置操作系统默认编码为 zh_CN.UTF-8
```bash
export LANG=zh_CN.UTF-8
```

#### Linux上安装elasticsearch出现错误 `java.lang.RuntimeException: can not run elasticsearch as root`

elasticsearch 需要在非root环境下运行，可以做如下的操作：

```
adduser est
chown est:est -R ${HOME}/elasticsearch-8.3.2/
cd ${HOME}/elasticsearch-8.3.2/
su est
./bin/elasticsearch
```

#### Mac OS上安装elasticsearch出现错误 `flood stage disk watermark [95%] exceeded on.... all indices on this node will be marked read-only`

elasticsearch默认达到95％就全都设置只读，可以腾出一部分空间出来再启动，或者修改 `config/elasticsearch.pyml`。
```
cluster.routing.allocation.disk.threshold_enabled: false
```

#### nltk_data加载失败的错误 `[nltk_data] Error loading punkt: [Errno 60] Operation timed out`

在命令行里面输入python,然后输入下面的命令进行下载：

```
import nltk
nltk.download('punkt')
```
如果下载还是很慢，可以手动[下载](https://github.com/nltk/nltk_data/tree/gh-pages/packages/tokenizers)，然后放入本地的`~/nltk_data/tokenizers`进行解压即可。

#### 服务端运行报端口占用的错误 `[Errno 48] error while attempting to bind on address ('0.0.0.0',8891): address already in use`

```
lsof -i:8891
kill -9 PID # PID为8891端口的进程
```

### faiss 安装上了但还是显示找不到faiss怎么办？

推荐您使用anaconda进行单独安装，安装教程请参考[faiss](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)

```
# CPU-only version
conda install -c pytorch faiss-cpu

# GPU(+CPU) version
conda install -c pytorch faiss-gpu
```
