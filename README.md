# Material_Curve_Analyzer

##### by ZYL

#### 开发日志：

* RC1:基于M3,为legend模块引入EasyOCR，合并axies\_reunion.py和axies\_API.py,优化legend标签的识别方案，是上传git的最初版本
* RC2:加入对linux的支持，转移开发环境到linux，加入requirements.txt以便于配置环境,修改了gitignore屏蔽更多类型的无用文件
* RC3:加入读取文件夹的功能，对于对数轴导致的报错问题暂时采用忽略的方式处理,为OCR功能加入CUDA加速，大幅提升了运行速度，修改了结果文件输出的格式为比赛要求的格式
* RC4:引入多线程设计（HyperThread V1和HyperPipeline V1）,大幅提升了运行速度，在测试机上运行时间缩短了约28%（4分10秒->3分03秒，works=4），修复了数轴识别匹配时无法正常识别负号的问题，为yolo部分引入了CUDA加速

#### 整体设计：

* 采用高度模块化设计，每个组件均可单独运行调试，方便进行改进
* 传统视觉与神经网络融合设计，保证了高准确性和运行效率
