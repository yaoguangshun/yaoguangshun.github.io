---
title: 物体检测中的anchor
date: 2019-3-26 17:55:00
tags: [object detection]
mathjax: true
categories: CV
---

anchor自从在faster rcnn中提出后，被广泛应用于物体检测中，流程可概括如下（one stage）

1. 通过backbone和neck（fpn等）网络得到feature map并根据feature map用head网络生成各个尺度的cls score和reg score
2. 生成anchor
3. 将anchor assign给对应的gt box
4. 根据assign result按一定策略sample出正负样本
5. 计算loss

（以上英文单词均遵循mmdetection的命名）

&emsp;&emsp;而anchor生成在一般的实现中都会使用到三个参数，anchor ratios，anchor strides，scales，其中anchor ratios很好解释，指的就是宽高比（h:s）；而anchor strides则有一点绕，我的理解是它指的是感受野的大小，也就是对应feature map对应到原图多少个像素，所以它的值等于原图中anchor的基础大小，这在代码中一般写作base_size，有了base_size，就可以得到一个基础框，此时一般使用((0, 0), (anchor_stride-1, anchor_stride-1))这个框，也就是图中最左上的一个框。。通过这个坐标，可以很轻松的计算出中心(x_ctr, y_ctr)，宽和高就是base_size；scales也很简单，就是放缩的尺度，要将宽和高按照各个scale的值放大。

&emsp;&emsp;例如：当anchor stride = 4，anchor_ratios = [0.5, 1.0, 2.0]，scales = [4, 8, 16]时，首先可以得到一个((0,0),(3,3))的基础框，宽高都为4，将这个4$\times$4的区域按照anchor_ratios进行变换，保证面积不变，宽高比符合要求，可以做如下转换
$$
\begin{align}
& hs=h*\sqrt{ratio} \\
& ws = \displaystyle\frac{w}{\sqrt{ratio}}
\end{align}
$$

也可以按如下方式转换
$$
\begin{align}
& hs = \sqrt{wh\cdot ratio} \\
& ws = \displaystyle\frac{\sqrt{wh}}{\sqrt{ratio}}
\end{align}
$$
&emsp;&emsp;无论用哪种方式都能保证面积不变，而宽高比符合anchor_ratios。将转换后的hs和ws乘上对应的scales，就能得到放缩后的anchor，有三种ratio，三种scale，一共可以得到9个anchor，再根据中心坐标，算出每个anchor的((xmin, ymin),(xmax, ymax))。

&emsp;&emsp;得到左上角的所有anchor坐标之后，只需要对他们进行坐标的平移，就能得到整张图片上所有的anchor的坐标，有平移就会涉及到x和y方向的stride，一般将stride设为与anchor_stride相等，也可以根据情况自己设置。

&emsp;&emsp;mmdetection中AnchorGenerator类实现的就是这一过程，除此之外，如果坐标在图外显然是invalid的，mmdetection中还对feat_map上有效的anchor进行了计算，该类中仅仅处理了右侧和下侧的边界问题，而对于上侧和左侧的边界，也就是0一侧的判断是在`anchor_target.py`中的`anchor_inside_flags`函数中处理的，该函数中同样考虑了右侧和下侧的边界，得到的是所有合法anchor。

以下是AnchorGenerator的代码，结合上面的说明理解起来并不困难

```python
class AnchorGenerator(object):

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        h_ratios = torch.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).view(-1)

        base_anchors = torch.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            dim=-1).round()

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = self.base_anchors.to(device)

        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(
            valid.size(0), self.num_base_anchors).contiguous().view(-1)
        return valid
```



