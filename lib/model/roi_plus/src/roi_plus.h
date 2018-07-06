int roi_plus_forward(int aligned_height, int aligned_width, float spatial_scale,
                      THFloatTensor * features, THFloatTensor * rois, THFloatTensor * output);

int roi_plus_backward(int aligned_height, int aligned_width, float spatial_scale,
                      THFloatTensor * top_grad, THFloatTensor * rois, 
                      THFloatTensor * bottom_grad, THFloatTensor * offset_x_width,
                      THFloatTensor * offset_y_width, THFloatTensor * feature_data);
