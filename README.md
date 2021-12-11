# Pocket GLUE in Torch

NLP做分类任务时常需要提交自己的算法到GLUE进行评测，可是就能找到的评测代码而言，它们不太容易读懂和修改（毕竟耦合了其他很多操作）。因此，就打算写一个袖珍版的GLUE评测代码，供大家使用。

GLUE benchmark: [General Language UnderstandingEvaluation](https://gluebenchmark.com/)

  

### ⭐️欢迎star和提问～

  
### 使用

- 下载[GLUE数据集](https://gluebenchmark.com/)，若不方便可从以下链接下载：

  |百度网盘    |  [链接](https://pan.baidu.com/s/1Q4UyQW8AVR1smRxnEhFFKg)|提取码: 94jy |
  | --- | --- | --- |
  |  谷歌云盘   |   [链接]()  |     |


- 下载bert预训练的权重(这里使用的是[HuggingFace的bert](https://huggingface.co/bert-base-uncased))到指定文件夹；

- 代码组成。运行时配置好超参和环境之后直接`python main.py`即可
  ```
  ├── main.py
  ├── config.py ## 配置超参数  
  ├── load_data.py ## 加载GLUE的数据
  ├── helper.py ## 设置随机种子
  ├── requirements.txt ## 环境
  ```
### 交流可加微信：Yunpengtai
