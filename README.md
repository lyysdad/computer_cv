# computer_cv_work
# 基于DCT和Squeeze and Excitation的光刻热点检测
benchmarks是经过dct压缩后的数据集
##
ini是训练数据集的配置文件

启动步骤

conda create --name hsd python=3.6

conda activate hsd

pip install -r requirements.txt

cd train

编辑修改 name = “” 例如 name = "iccad2_config"

进行训练

python train_cnn_focal_loss.py

训练完后模型保存在models/iccad2/lzh下

修改iccad2_config.ini中的刚训练完得到的模型路径 model_path = ../models/iccad2/lzh/model-9999-0.3-focal_loss.ckpt

测试

cd test

python test_cnn_focal_loss.py

log下存放训练日志和测试日志
