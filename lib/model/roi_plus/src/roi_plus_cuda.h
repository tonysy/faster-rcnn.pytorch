int roi_plus_forward_cuda(int aligned_height, int aligned_width, float spatial_scale,
                        THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output);

int roi_plus_backward_cuda(int aligned_height, int aligned_width, float spatial_scale,
                        THCudaTensor * top_grad, THCudaTensor * rois,
                        THCudaTensor * bottom_grad, THCudaTensor * offset_x_grad,
                        THCudaTensor * offset_y_grad, THCudaTensor * features);
