import os
import sys

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['E:\\DeepLearn\\NetModel', 'E:\\DeepLearn\\NetModel\\SequenceModel', 'E:\\DeepLearn\\NetModel\\NGrams',
                 'E:\\DeepLearn\\NetModel\\RNN', 'E:\\DeepLearn\\NetModel\\CIFAR-10',
                 'E:\\DeepLearn\\NetModel\\FineTuning', 'E:\\DeepLearn\\NetModel\\ImageNet-Dogs',
                 'E:\\DeepLearn\\NetModel\\NiN', 'E:\\DeepLearn\\NetModel\\GoogLeNet', 'E:\\DeepLearn\\NetModel\\ResNet',
                 'E:\\DeepLearn\\NetModel\\DenseNet', 'E:\\DeepLearn\\NetModel\\AlexNet',
                 'E:\\DeepLearn\\NetModel\\LeNet', 'E:\\DeepLearn\\NetModel\\VGG', 'E:\\DeepLearn\\NetModel\\Seq2Seq',
                 'E:\\DeepLearn\\NetModel\\Attention', 'E:/DeepLearn/NetModel'])

# 输入 子目录 运行文件名
print(sys.argv)
os.chdir(sys.argv[1])
with open(sys.argv[2], encoding="utf-8") as f:
    exec(f.read())
