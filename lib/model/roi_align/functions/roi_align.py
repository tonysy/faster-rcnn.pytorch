import torch
from torch.autograd import Function
from .._ext import roi_align


# TODO use save_for_backward instead
class RoIAlignFunction(Function):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        self.rois = rois # e.g. shape [128, 5]
        self.feature_size = features.size() # e.g. shape [1, 1024, 25, 33]

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0) # e.g. 128

        output = features.new(num_rois, num_channels, self.aligned_height, self.aligned_width).zero_()
        # e.g. output.shape  [128, 1024, 8, 8]

        if features.is_cuda:
            roi_align.roi_align_forward_cuda(self.aligned_height,
                                             self.aligned_width,
                                             self.spatial_scale, features,
                                             rois, output)
        else:
            roi_align.roi_align_forward(self.aligned_height,
                                        self.aligned_width,
                                        self.spatial_scale, features,
                                        rois, output)
#            raise NotImplementedError

        return output

    def backward(self, grad_output):
        # e.g. grad_output.shape [128, 1024, 8, 8]

        assert(self.feature_size is not None and grad_output.is_cuda)

        # e.g. [1, 1024, 25, 33]
        batch_size, num_channels, data_height, data_width = self.feature_size

        grad_input = self.rois.new(batch_size, num_channels, data_height,
                                  data_width).zero_()
        roi_align.roi_align_backward_cuda(self.aligned_height,
                                          self.aligned_width,
                                          self.spatial_scale, grad_output,
                                          self.rois, grad_input)

        # print grad_input

        return grad_input, None
