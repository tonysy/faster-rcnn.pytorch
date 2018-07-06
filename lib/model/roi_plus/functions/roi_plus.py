import torch
from torch.autograd import Function
# from .._ext import roi_align
from .._ext import roi_plus


# TODO use save_for_backward instead
class RoIPlusFunction(Function):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        self.rois = rois
        self.feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, self.aligned_height, self.aligned_width).zero_()
        if features.is_cuda:
            roi_plus.roi_plus_forward_cuda(self.aligned_height,
                                             self.aligned_width,
                                             self.spatial_scale, features,
                                             rois, output)
        else:
            roi_plus.roi_plus_forward(self.aligned_height,
                                        self.aligned_width,
                                        self.spatial_scale, features,
                                        rois, output)
#            raise NotImplementedError
        self.num_rois = num_rois
        self.feature = features

        return output

    def backward(self, grad_output):
        assert(self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = self.feature_size
        # import pdb; pdb.set_trace()
        grad_input = self.rois.new(batch_size, num_channels, data_height,
                                  data_width).zero_()

        offset_x_grad = self.rois.new(self.num_rois, num_channels, self.aligned_height, self.aligned_width).zero_()
        offset_y_grad = self.rois.new(self.num_rois, num_channels, self.aligned_height, self.aligned_width).zero_()

        roi_plus.roi_plus_backward_cuda(self.aligned_height,
                                          self.aligned_width,
                                          self.spatial_scale, grad_output,
                                          self.rois, grad_input,
                                          offset_x_grad, offset_y_grad,
                                          self.feature)

        offset_x_grad = torch.sum(torch.sum(offset_x_grad, -1), -1)
        offset_y_grad = torch.sum(torch.sum(offset_y_grad, -1), -1)
        offset_grad = torch.cat((offset_x_grad.unsqueeze(-1),offset_y_grad.unsqueeze(-1)),-1)
        import pdb; pdb.set_trace()

        # print grad_input
        # import pdb; pdb.set_trace()

        # return grad_input, None
        return grad_input, offset_grad
