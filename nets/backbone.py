import torch
import torch.nn as nn


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p

class SiLU(nn.Module):  
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=SiLU()):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    
class Block(nn.Module):
    def __init__(self, c1, c2, c3, n=4, e=1, ids=[0]):
        super(Block, self).__init__()
        c_ = int(c2 * e)
        
        self.ids = ids
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = nn.ModuleList(
            [Conv(c_ if i ==0 else c2, c2, 3, 1) for i in range(n)]
        )
        self.cv4 = Conv(c_ * 2 + c2 * (len(ids) - 2), c3, 1, 1)

    def forward(self, x):
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)
        
        x_all = [x_1, x_2]
        for i in range(len(self.cv3)):
            x_2 = self.cv3[i](x_2)
            x_all.append(x_2)
            
        out = self.cv4(torch.cat([x_all[id] for id in self.ids], 1))
        return out

class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)
    
class Transition(nn.Module):
    def __init__(self, c1, c2):
        super(Transition, self).__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.cv3 = Conv(c2, c2, 3, 2)
        
        self.mp  = MP()

    def forward(self, x):
        x_1 = self.mp(x)
        x_1 = self.cv1(x_1)
        
        x_2 = self.cv2(x)
        x_2 = self.cv3(x_2)
        
        return torch.cat([x_2, x_1], 1)

#  ====================================== RCAG ===============================================
class RCAG(nn.Module):
    def __init__(self, F_int, rcag_):
        super(RCAG, self).__init__()

        self.conv_g = nn.Conv2d(F_int, F_int, kernel_size=(1, 1))
        self.conv_f = nn.Conv2d(F_int, F_int, kernel_size=(1, 1))
        self.relu1 = nn.ReLU()

        self.conv_h = nn.Conv2d(F_int, F_int*2, kernel_size=(1, 1))
        self.GAP = nn.AdaptiveAvgPool2d(rcag_)

        self.mlp = nn.Linear(rcag_, int(F_int))
        # self.mlp_relu = nn.ReLU()
        self.mlp2 = nn.Linear(int(F_int), rcag_)
        self.attention_map = nn.Sigmoid()

    def forward(self, g, x):

        p = self.conv_f(g)
        q = self.conv_g(x)
        add_1 = p + q
        z = self.relu1(add_1)
        z_in = self.conv_h(z)
        res = z_in

        x = self.GAP(z_in)
        x = self.mlp(x)
        x = self.mlp2(x)
        x = self.attention_map(x)

        x = res*x

        x += res

        return x
class ATTention(nn.Module):
    def __init__(self, F_int, rcag_):
        super(ATTention, self).__init__()

        self.GAP = nn.AdaptiveAvgPool2d(rcag_)

        self.mlp = nn.Linear(rcag_, int(F_int))
        # self.mlp_relu = nn.ReLU()
        self.mlp2 = nn.Linear(int(F_int), rcag_)
        self.attention_map = nn.Sigmoid()

    def forward(self, x):

        res = x
        x = self.GAP(x)
        x = self.mlp(x)
        x = self.mlp2(x)
        x = self.attention_map(x)

        x = res*x
        x += res
        return x
#  ======================================== RCAG =============================================

#  ======================================== simam =============================================
class simam_module(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()
 
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
 
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s
 
    @staticmethod
    def get_module_name():
        return "simam"
 
    def forward(self, x):
 
        b, c, h, w = x.size()
        
        n = w * h - 1
 
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
 
        return x * self.activaton(y)
#  ======================================== simam =============================================

#  ======================================== cbam ==============================================
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        #使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)
 
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        #map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out
 
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()
 
    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
#  ======================================== cbam ==============================================
    
class Backbone(nn.Module):
    def __init__(self, transition_channels, block_channels, n, phi, pretrained=False):
        super().__init__()
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #-----------------------------------------------#
        ids = {
            'l' : [-1, -3, -5, -6],
            'x' : [-1, -3, -5, -7, -8], 
        }[phi]
        self.stem = nn.Sequential(
            Conv(3, transition_channels, 3, 1),
            Conv(transition_channels, transition_channels * 2, 3, 2),
            Conv(transition_channels * 2, transition_channels * 2, 3, 1),
        )
        self.dark2 = nn.Sequential(
            Conv(transition_channels * 2, transition_channels * 4, 3, 2),
            Block(transition_channels * 4, block_channels * 2, transition_channels * 8, n=n, ids=ids),
        )
        self.dark3 = nn.Sequential(
            Transition(transition_channels * 8, transition_channels * 4),
            Block(transition_channels * 8, block_channels * 4, transition_channels * 16, n=n, ids=ids),
        )
        self.dark4 = nn.Sequential(
            Transition(transition_channels * 16, transition_channels * 8),
            Block(transition_channels * 16, block_channels * 8, transition_channels * 32, n=n, ids=ids),
        )
        self.dark5 = nn.Sequential(
            Transition(transition_channels * 32, transition_channels * 16),
            Block(transition_channels * 32, block_channels * 8, transition_channels * 32, n=n, ids=ids),
        )
        
        if pretrained:
            url = {
                "l" : 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_backbone_weights.pth',
                "x" : 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        # feat0 = x   # ---new
        #-----------------------------------------------#
        #   dark3的输出为80, 80, 256，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        #-----------------------------------------------#
        #   dark4的输出为40, 40, 512，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        #-----------------------------------------------#
        #   dark5的输出为20, 20, 1024，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        feat3 = x

        # print(feat1.size())
        # print(feat2.size())
        # print(feat3.size())
        return feat1, feat2, feat3
        # return feat0, feat1, feat2, feat3
