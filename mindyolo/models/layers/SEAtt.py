import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import HeNormal, Constant, Normal

class SEAttention(nn.Cell):
    def __init__(self, channel=512, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.SequentialCell([
            nn.Dense(channel, channel // reduction, has_bias=False),
            nn.ReLU(),
            nn.Dense(channel // reduction, channel, has_bias=False),
            nn.Sigmoid()
        ])
        self.init_weights()

    def init_weights(self):
        # 初始化权重部分的修改
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                # 使用正确的权重初始化
                cell.weight.set_data(mindspore.common.initializer.initializer(HeNormal(mode='fan_out'),cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(mindspore.common.initializer.initializer("ones", cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(mindspore.common.initializer.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(mindspore.common.initializer.initializer("zeros", cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(mindspore.common.initializer.initializer(Normal(0.001), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(mindspore.common.initializer.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def construct(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).reshape(b, c)  # 替代 view
        y = self.fc(y).reshape(b, c, 1, 1)  # 替代 view
        return x * ops.BroadcastTo(x.shape)(y)  # 自动广播操作
