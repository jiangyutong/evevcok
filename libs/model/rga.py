import torch
from torch import nn
from torch.nn import functional as F

class RGA_Module(nn.Module):
    def __init__(self, in_channel, size=(32,32), use_spatial=True, use_channel=False,
                 cha_ratio=8, spa_ratio=8, down_ratio=8):
        super(RGA_Module, self).__init__()
        self.in_channel = in_channel
        if not isinstance(size, tuple):
            size = (size, size)
        self.size = size
        self.spatial_size = size[0] * size[1]

        self.use_spatial = use_spatial
        self.use_channel = use_channel

        self.inter_channel = in_channel // cha_ratio
        self.inter_spatial = self.spatial_size // spa_ratio

        self.adaptive_max_pool = nn.AdaptiveMaxPool2d(output_size=self.size)

        if self.use_spatial:
            # Embedding functions for modeling spatial relations
            self.spa_alpha = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU(inplace=True)
            )
            self.spa_beta = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU(inplace=True)
            )
            # Embedding functions for original features
            self.spa_embed_ori_feat = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU(inplace=True)
            )
            # Embedding functions for spatial relation features
            self.spa_embed_rel_feat = nn.Sequential(
                nn.Conv2d(in_channels=self.spatial_size * 2, out_channels=self.inter_spatial, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU(inplace=True)
            )
            # Networks for learning spatial attention weights
            start_channel = self.inter_spatial + 1
            mid_channel = start_channel // down_ratio
            self.spa_weights = nn.Sequential(
                nn.Conv2d(in_channels=start_channel, out_channels=mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=mid_channel, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )

        if self.use_channel:
            # Embedding functions for modeling channel relations
            self.cha_alpha = nn.Sequential(
                nn.Conv2d(in_channels=self.spatial_size, out_channels=self.inter_spatial, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU(inplace=True)
            )
            self.cha_beta = nn.Sequential(
                nn.Conv2d(in_channels=self.spatial_size, out_channels=self.inter_spatial, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU(inplace=True)
            )
            # Embedding functions for original features
            self.cha_embed_ori_feat = nn.Sequential(
                nn.Conv2d(in_channels=self.spatial_size, out_channels=self.inter_spatial, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
            # Embedding functions for channel relation features
            self.cha_embed_rel_feat = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel * 2, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
            # Networks for learning channel attention weights
            start_channel = self.inter_channel + 1
            mid_channel = start_channel // down_ratio
            self.cha_weights = nn.Sequential(
                nn.Conv2d(in_channels=start_channel, out_channels=mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_channel),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channel, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )

    def forward(self, x):
        b, c, h, w = x.shape
        fh, fw = self.size
        # The size of the input(x) can have different size, but the same channels.
        # It will be converted to a fixed size by nn.AdaptiveMaxPool2d()
        if self.use_spatial:
            fixed_x = self.adaptive_max_pool(x)  # [b, c, fh, fw]. Making the input x to be a fixed size
            alpha_x = self.spa_alpha(fixed_x)    # Embedding functions for modeling spatial relations
            beta_x = self.spa_beta(fixed_x)      # Embedding functions for modeling spatial relations
            alpha_x = alpha_x.view(b, self.inter_channel, -1).permute(0, 2, 1)
            beta_x = beta_x.view(b, self.inter_channel, -1)
            pr = torch.matmul(alpha_x, beta_x)
            pr_in = pr.permute(0, 2, 1).view(b, fh * fw, fh, fw)
            pr_out = pr.view(b, fh * fw, fh, fw)
            pr_cat = torch.cat((pr_in, pr_out), dim=1)
            pr_cat = self.spa_embed_rel_feat(pr_cat)        # Embedding functions for spatial relation features

            o_x = self.spa_embed_ori_feat(fixed_x)          # Embedding functions for original features
            o_x = torch.mean(o_x, dim=1, keepdim=True)
            py = torch.cat((o_x, pr_cat), 1)

            spa_w = self.spa_weights(py)                    # Networks for learning spatial attention weights
            spa_w = F.interpolate(spa_w, size=(h, w), mode='nearest')
            if not self.use_channel:
                out = torch.sigmoid(spa_w) * x
                return out
            else:
                x = torch.sigmoid(spa_w) * x

        if self.use_channel:
            fixed_x = self.adaptive_max_pool(x)                          # [b, c, fh, fw]. Making the input x to be a fixed size
            xc = fixed_x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)
            alpha_xc = self.cha_alpha(xc).squeeze(-1).permute(0, 2, 1)   # Embedding functions for modeling channel relations
            beta_xc = self.cha_beta(xc).squeeze(-1)                      # Embedding functions for modeling channel relations
            cr = torch.matmul(alpha_xc, beta_xc)
            cr_in = cr.permute(0, 2, 1).unsqueeze(-1)
            cr_out = cr.unsqueeze(-1)
            cr_cat = torch.cat((cr_in, cr_out), dim=1)
            cr_cat = self.cha_embed_rel_feat(cr_cat)                     # Embedding functions for channel relation features

            o_cx = self.cha_embed_ori_feat(xc)
            o_cx = torch.mean(o_cx, dim=1, keepdim=True)
            cy = torch.cat((o_cx, cr_cat), dim=1)

            cha_w = self.cha_weights(cy).transpose(1,2)                  # Networks for learning channel attention weights
            out = torch.sigmoid(cha_w) * x

            return out

    def weight_mapping(self):
        weight_mapping = {
            name:'_'.join(name.split('.')) for name,_ in self.named_parameters()
        }
        return weight_mapping

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    rga = RGA_Module(in_channel=1024, size=20, use_spatial=True, use_channel=True).to(device)

    # query_rga = RGA_Module(in_channel=1024, size=32, use_spatial=True, use_channel=True).to(device)
    # support_rag = RGA_Module(in_channel=1024, size=20, use_spatial=True, use_channel=True).to(device)
    batch_size = 2
    way, shot = 2, 5
    channel = 1024
    query = torch.rand((batch_size, channel, 38, 57)).to(device)
    support = torch.rand((batch_size * way * shot, channel, 20, 20)).to(device)

    q = rga(query)
    s = rga(support)

    a = 0