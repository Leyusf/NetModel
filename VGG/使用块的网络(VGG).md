VGG通过使用循环和子程序，可以很容易地在任何现代深度学习框架的代码中实现这些重复的架构。

## VGG块
由一系列卷积层组成，后面再加上用于空间下采样的最大汇聚层。该函数有三个参数，分别对应于卷积层的数量`num_convs`、输入通道的数量`in_channels` 和输出通道的数量`out_channels`。
```
class VGGBlock:  
    def __init__(self, num_conv, in_channels, out_channels):  
        """  
        :param num_conv: 卷积层数量  
        :param in_channels: 输入通道数  
        :param out_channels: 输出通道数  
        """        layers = []  
        for _ in range(num_conv):  
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1))  
            layers.append(nn.ReLU())  
            in_channels = out_channels  
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  
        self.block = nn.Sequential(*layers)
```

## VGG网络
![[Pasted image 20231118134810.png|575]](../images/20231118134810.png)

VGG网络由几个VGG块组成。


