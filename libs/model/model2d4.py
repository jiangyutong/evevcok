"""
Fully-connected residual network as a single deep learner.
"""
import torch.nn as nn
import math

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torch.nn as nn
import torch



class ResidualBlock(nn.Module):
    """
    A residual block.
    """

    def __init__(self, linear_size, p_dropout=0.5, kaiming=True, leaky=False):
        super(ResidualBlock, self).__init__()
        self.l_size = linear_size
        if leaky:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)
        if kaiming:
            self.w1.weight.data = nn.init.kaiming_normal_(self.w1.weight.data)
            self.w2.weight.data = nn.init.kaiming_normal_(self.w2.weight.data)
        # # 网络的第一层加入注意力机制
        # self.ca = ChannelAttention(self.l_size)
        # self.sa = SpatialAttention()
        # # 网络的卷积层的最后一层加入注意力机制
        # self.ca1 = ChannelAttention(self.l_size)
        # self.sa1 = SpatialAttention()

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 17, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 17, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x=x.unsqueeze(dim=2).unsqueeze(dim=3)
        avg_out=self.avg_pool(x)
        avg_out=self.fc1(avg_out)
        avg_out=self.relu1(avg_out)
        avg_out = self.fc2(avg_out)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = out.squeeze()
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(34, 34, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

class FCModel(nn.Module):
    def __init__(self,
                 linear_size=1024,
                 num_blocks=2,
                 p_dropout=0.5,
                 norm_twoD=False,
                 kaiming=False,
                 refine_3d=False,
                 leaky=False,
                 dm=False,
                 input_size=64,
                 output_size=64):
        """
        Fully-connected network.
        """
        super(FCModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_blocks = num_blocks
        self.refine_3d = refine_3d
        self.leaky = leaky
        self.dm = dm
        self.input_size = input_size
        self.output_size = output_size

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.w11 = nn.Linear(self.input_size, self.input_size)
        self.w12 = nn.Linear(self.input_size, self.output_size)
        self.w13 = nn.Linear(self.linear_size, self.output_size)
        self.w14 = nn.Linear(self.output_size, self.output_size)
        self.batch_norm11 = nn.BatchNorm1d(self.input_size)
        self.batch_norm12 = nn.BatchNorm1d(self.linear_size)
        self.batch_norm13 = nn.BatchNorm1d(self.output_size)
        # self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
        # self.res_blocks=ResidualBlock(self.linear_size,
        #                                      self.p_dropout,
        #                                      leaky=self.leaky)
        # output
        # self.w2 = nn.Linear(self.linear_size, self.output_size)
        # rnn = nn.LSTM(10, 20, 2)
        # 输入数据x的向量维数10, 设定lstm隐藏层的特征维度20, 此model用2个lstm层。如果是1，可以省略，默认为1)
        self.lstm = nn.LSTM(2, 2,batch_first=True,num_layers=4)

        if self.leaky:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        if kaiming:
            self.w11.weight.data = nn.init.kaiming_normal_(self.w11.weight.data)
            self.w12.weight.data = nn.init.kaiming_normal_(self.w12.weight.data)

        self.ca = ChannelAttention(input_size)
        self.sa = SpatialAttention()
    def forward(self, x):
        y = self.ca(x) * x
        # # y = self.sa(y) * y
        y = self.w12(y)
        y = y + x
        y3 = self.w11(x)
        # # y3 = self.batch_norm11(y3)
        # # y3 = self.relu(y3)
        # # y3 = self.dropout(y3)
        y3=y3+x
        y3 = self.w11(y3)
        # # y3 = self.batch_norm11(y3)
        # # y3 = self.relu(y3)
        # # y3 = self.dropout(y3)
        y3 = y3 + x
        # y3 = self.w11(y3)
        # y3 = self.batch_norm11(y3)
        # y3 = self.relu(y3)
        # y3 = self.dropout(y3)
        # y = self.w12(y)
        #
        # y2=self.w1(x)
        # y2 = self.batch_norm12(y2)
        # y2 = self.relu(y2)
        # y2 = self.dropout(y2)
        # y2 =self.w13(y2)
        # y=y+y2
        # y=
        # input(batch, seq_len,input_size)
        y2=y.reshape(-1,17,2)
        # y2=y2.transpose(1, 2)
        # h_0(num_layers * num_directions, batch, hidden_size)
        h0 = torch.randn(4, y2.shape[0], 2).cuda()

        c0 = torch.randn(4, y2.shape[0], 2).cuda()
        y2 = self.lstm(y2,(h0,c0))

        # y2 = y2[0].transpose(1, 2)
        y2 = y2[0].reshape(-1,34)
        # y2 = self.batch_norm11(y2)
        # y2 = self.relu(y2)
        # y2 = self.dropout(y2)
        y4=y2+y3
        # y2 = y4.reshape(-1, 17, 2)
        # # y2=y2.transpose(1, 2)
        # # h_0(num_layers * num_directions, batch, hidden_size)
        # y2 = self.lstm(y2, (h0, c0))
        # # y2 = y2[0].transpose(1, 2)
        # y2 = y2[0].reshape(-1, 34)
        # y2 = self.batch_norm11(y2)
        # y2 = self.relu(y2)
        # y2 = self.dropout(y2)
        # y4= y2 + y4
        # y2 = y4.reshape(-1, 17, 2)
        # # y2=y2.transpose(1, 2)
        # # h_0(num_layers * num_directions, batch, hidden_size)
        # y2 = self.lstm(y2, (h0, c0))
        # # y2 = y2[0].transpose(1, 2)
        # y2 = y2[0].reshape(-1, 34)
        # # y2 = self.batch_norm11(y2)
        # # y2 = self.relu(y2)
        # # y2 = self.dropout(y2)
        # y2 = y2 + y4
        # y2 = y2.reshape(-1, 32, 2)
        # y2 = y2.transpose(1, 2)
        # y2 = self.lstm(y2, (h0, c0))
        # y2 = y2[0].transpose(1, 2)
        # y2 = y2.reshape(-1, 64)
        # y2 = self.batch_norm11(y2)
        # y2 = self.relu(y2)
        # y2 = self.dropout(y2)
        # y2 = y2 + x
        # y2 = y2.reshape(-1, 32, 2)
        # y2 = y2.transpose(1, 2)
        # # h_0(num_layers * num_directions, batch, hidden_size)
        # y2 = self.lstm(y2, (h0, c0))
        # y2 = y2[0].transpose(1, 2)
        # y2 = y2.reshape(-1, 64)
        # y2 = self.batch_norm11(y2)
        # y2 = self.relu(y2)
        # y2 = self.dropout(y2)
        # y2 = y2 + x
        # y2 = self.batch_norm11(y2)
        # y2 = self.relu(y2)
        # y2 = self.dropout(y2)
        # y = self.w12(x)
        # y = self.batch_norm11(y)
        # y = self.relu(y)
        # y = self.dropout(y)
        # y = y + x
        # y = self.w12(y)
        # y = self.batch_norm13(y)
        # y = self.relu(y)
        # y = self.dropout(y)
        y1= self.w1(y4)
        # y1 = self.batch_norm12(y1)
        # y1 = self.relu(y1)
        # y1 = self.dropout(y1)

        y1=self.w13(y1)
        # y1 = self.batch_norm13(y1)
        # y1 = self.relu(y1)
        # y1 = self.dropout(y1)
        y =  y3+y1
        # y = self.w14(y)
        # y = self.batch_norm13(y)
        # y = self.relu(y)
        # y = self.dropout(y)
        y = self.w12(y)
        # y = self.batch_norm13(y)
        # y = self.relu(y)
        # y = self.dropout(y)


        return y





def get_model(
              refine_3d=False,
              norm_twoD=False,
              num_blocks=2,
              input_size=64,
              output_size=64,
              linear_size=1024,
              dropout=0.5,
              leaky=False
              ):
    model = FCModel(
                    refine_3d=refine_3d,
                    norm_twoD=norm_twoD,
                    num_blocks=num_blocks,
                    input_size=input_size,
                    output_size=output_size,
                    linear_size=linear_size,
                    p_dropout=dropout,
                    leaky=leaky
                    )
    return model


def prepare_optim(model, opt):
    """
    Prepare optimizer.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if opt.optim_type == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr=opt.lr,
                                     weight_decay=opt.weight_decay
                                     )
    elif opt.optim_type == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    lr=opt.lr,
                                    momentum=opt.momentum,
                                    weight_decay=opt.weight_decay
                                    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=opt.milestones,
                                                     gamma=opt.gamma)
    return optimizer, scheduler


def get_cascade():
    """
    Get an empty cascade.
    """
    return nn.ModuleList([])