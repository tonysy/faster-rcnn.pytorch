from torch.nn.modules.module import Module
from torch.nn.functional import avg_pool2d, max_pool2d
# from ..functions.roi_align import RoIAlignFunction
from ..functions.roi_plus import RoIPlusFunction


class RoIPlus(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIPlus, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIPlusFunction(self.aligned_height, self.aligned_width,
                                self.spatial_scale)(features, rois)

class RoIPlusAvg(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIPlusAvg, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x =  RoIPlusFunction(self.aligned_height+1, self.aligned_width+1,
                                self.spatial_scale)(features, rois)
        return avg_pool2d(x, kernel_size=2, stride=1)

class RoIPlusMax(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIPlusMax, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x =  RoIPlusFunction(self.aligned_height+1, self.aligned_width+1,
                                self.spatial_scale)(features, rois)
        return max_pool2d(x, kernel_size=2, stride=1)
