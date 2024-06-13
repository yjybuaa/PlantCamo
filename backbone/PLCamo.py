import torch
import torch.nn as nn
import torch.nn.functional as F
# import backbone.resnet as resnet
from lib.pvtv2 import pvt_v2_b2





class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out

    def initialize(self):
        pass

class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(x) + p1_dc #+ self.p2_channel_reduction(x)*p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(x) + p2_dc #+ self.p3_channel_reduction(x)*p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc #+ self.p4_channel_reduction(x)*p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        return ce

class Focus(nn.Module):
    def __init__(self, channel1, channel2):
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.up2 = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU())
        #CBRU  文中的Fup

        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())
        self.input_map2 = nn.Sequential(nn.Sigmoid())
        #输入

        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)
        #输出

        self.fp = Context_Exploration_Block(self.channel1)
        #通过CE模块

        self.fn = Context_Exploration_Block(self.channel1)
        #通过CE模块

        self.alpha = nn.Parameter(torch.ones(1))
        #参数α

        self.beta = nn.Parameter(torch.ones(1))
        #参数β

        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()

    def forward(self, x, y, in_map,upsample = True):
        # x; current-level features
        # y: higher-level features
        # in_map: higher-level prediction

        if upsample :
            up = self.up(y)
        #首先输入高级别的特征y 通过CBRU（卷积 归一化 Relu 上采样）后输出
            input_map = self.input_map(in_map)
        #输入高级别预测  通过上采样和sigmoid函数 变化成0~1之间的数输出
        else:
            up = self.up2(y)
            input_map = self.input_map2(in_map)
        f_feature = x * input_map
        #假阳性特征  由当前特征x*输入 表示

        b_feature = x * (1 - input_map)
        #假阴性特征 由当前特征x*（1-输入）表示

        fp = self.fp(f_feature)
        #将假阳性特征输入到CE上下文模块中 得到输出fpd

        fn = self.fn(b_feature)
        #将假阴性特征输入到CE上下文模块中 得到输出fnd

        refine1 = up - (self.alpha * fp)
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)
        #Fr = BR(Fup−αFfpd)

        refine2 = refine1 + (self.beta * fn)
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)
        #F′r = BR(Fr +βFf nd)

        output_map = self.output_map(refine2)
        #卷积后的输出

        return refine2, output_map


class Att(nn.Module):
    def __init__(self, channels=64, r=4):
        super(Att, self).__init__()
        out_channels = int(channels // r)
        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sig(xlg)

        return wei

    def initialize(self):
        # weight_init(self)
        pass




class MyNet(nn.Module):
    def __init__(self, channel = 32) :
        super(MyNet,self).__init__()

        #backbone 的加载
        self.backbone = pvt_v2_b2()
        path = './pretrained_pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)


        self.cr4 = nn.Sequential(nn.Conv2d(512,64,1), nn.BatchNorm2d(64), nn.ReLU())
        self.cr3 = nn.Sequential(nn.Conv2d(320,64,1), nn.BatchNorm2d(64), nn.ReLU())
        self.cr2 = nn.Sequential(nn.Conv2d(128,64,1), nn.BatchNorm2d(64), nn.ReLU())
        self.cr1 = nn.Sequential(nn.Conv2d(64,64,1), nn.BatchNorm2d(64), nn.ReLU())

        self.conv0 = nn.Sequential(self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], 64),nn.BatchNorm2d(64), nn.ReLU())
        self.conv1 = nn.Sequential(self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], 64),nn.BatchNorm2d(64), nn.ReLU())
        self.conv2 = nn.Sequential(self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], 64),nn.BatchNorm2d(64), nn.ReLU())
        self.conv3 = nn.Sequential(self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], 64),nn.BatchNorm2d(64), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64,64,3,1,1,1), nn.BatchNorm2d(64), nn.ReLU())
        
        self.cbr1 = nn.Sequential(nn.Conv2d(64,64,3,1,1,1), nn.BatchNorm2d(64), nn.ReLU())
        self.cbr2 = nn.Sequential(nn.Conv2d(64,64,3,1,1,1), nn.BatchNorm2d(64), nn.ReLU())
        self.cbr3 = nn.Sequential(nn.Conv2d(64,64,3,1,1,1), nn.BatchNorm2d(64), nn.ReLU())
        self.cbr4 = nn.Sequential(nn.Conv2d(64,64,3,1,1,1), nn.BatchNorm2d(64), nn.ReLU())
        

        self.Att0 = Att()
        self.Att1 = Att()
        self.Att2 = Att()
        self.Att3 = Att()
        self.Att4 = Att()


        self.map = nn.Conv2d(64, 1, 7, 1, 3)

        self.fm1 = Focus(64,64)
        self.fm2 = Focus(64,64)
        self.fm3 = Focus(64,64)
        self.fm4 = Focus(64,64)

        self.out_map = nn.Conv2d(64, 1, 7, 1, 3)


        

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True
        
    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)

    def forward(self, x):
        """
            x : 3  * 448 * 448
            x1: 64 * 176 * 176
            x2: 128* 88  * 88
            x3: 320* 44  * 44
            x4: 512* 22  * 22

        """

        #提取特征
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        x_4 = self.cr4(x4)  #64
        x_3 = self.cr3(x3)  #64
        x_2 = self.cr2(x2)  #64
        x_1 = self.cr1(x1)  #64

        

        

############################      Iterate  Block  ##################
        stage_loss1=list()
        stage_loss2=list()
        FeedBack_feature=None
        for iter in range(2):
            if FeedBack_feature==None:
                f_1 =x_1     #64
            else:
                f_1 =x_1 + FeedBack_feature    #
                
            f_2 = x_2           #64
            f_3 = x_3           #64
            f_4 = x_4           #64

            gf0 = f_1
            gf0 = self.conv0(f_1)
            gf0 = gf0*self.Att0(gf0)
            gf0 = F.interpolate(gf0, size=x_2.size()[2:], mode='bilinear')

            # gf1 = gf0+f_2
            gf1 = self.conv1(gf0+f_2)
            gf1 = gf1*self.Att1(gf1)
            gf1 = F.interpolate(gf1, size=x_3.size()[2:], mode='bilinear')

            # gf2 =gf1+f_3
            gf2 = self.conv2(gf1+f_3)
            gf2 = gf2*self.Att2(gf2)
            gf2 = F.interpolate(gf2, size=x_4.size()[2:], mode='bilinear')

            # gf3 =gf2+f_4
            gf3 = self.conv3(gf2+f_4)
            gf3 = gf3*self.Att3(gf3)
            gf3 = self.conv4(gf3)
            gf_pre = self.map(gf3)

            

            # rf4,rf4_map = self.fm4(f_4,gf3,gf_pre,upsample = False)
            # rf3,rf3_map = self.fm3(f_3,rf4,rf4_map)
            # rf2,rf2_map = self.fm2(f_2,rf3,rf3_map)
            # rf1,rf1_map = self.fm1(f_1,rf2,rf2_map)
            rf4 = f_4 + gf3
            rf4 = self.cbr1(rf4)
            rf4 = F.interpolate(rf4, size=x_3.size()[2:], mode='bilinear')
            rf3 = rf4 + f_3
            rf3 = self.cbr2(rf3)
            rf3 = F.interpolate(rf3, size=x_2.size()[2:], mode='bilinear')
            rf2 = rf3 + f_2
            rf2 = self.cbr3(rf2)
            rf2 = F.interpolate(rf2, size=x_1.size()[2:], mode='bilinear')
            rf1 = rf2 + f_1
            rf1 = self.cbr4(rf1)
            rf1_map = self.out_map(rf1)

            FeedBack_feature = rf1

            gf_pre = F.interpolate(gf_pre, size=x.size()[2:], mode='bilinear')
            rf1_map= F.interpolate(rf1_map, size=x.size()[2:], mode='bilinear')
            stage_loss1.append(gf_pre)
            stage_loss2.append(rf1_map)
           

        
        return stage_loss1,stage_loss2

