scripts：运行数据处理脚本目录, 将官方公开数据集转换成模型所需训练数据格式
运行命令：
  python run_build_data.py [udc|swda|mrda|atis|dstc2]

1)、生成MATCHING任务所需要的训练集、开发集、测试集时:
python run_build_data.py udc
生成数据在dialogue_general_understanding/data/input/data/udc

2)、生成DA任务所需要的训练集、开发集、测试集时:
  python run_build_data.py swda
  python run_build_data.py mrda
  生成数据分别在dialogue_general_understanding/data/input/data/swda和dialogue_general_understanding/data/input/data/mrda

3)、生成DST任务所需的训练集、开发集、测试集时:
  python run_build_data.py dstc2
  生成数据分别在dialogue_general_understanding/data/input/data/dstc2

4)、生成意图解析, 槽位识别任务所需训练集、开发集、测试集时:
  python run_build_data.py atis
  生成槽位识别数据在dialogue_general_understanding/data/input/data/atis/atis_slot
  生成意图识别数据在dialogue_general_understanding/data/input/data/atis/atis_intent
