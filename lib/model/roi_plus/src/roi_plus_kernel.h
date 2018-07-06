#ifndef _ROI_PLUS_KERNEL
#define _ROI_PLUS_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

__global__ void ROIPlusForward(const int nthreads, const float* bottom_data,
    const float spatial_scale, const int height, const int width,
    const int channels, const int aligned_height, const int aligned_width,
    const float* bottom_rois, float* top_data);

int ROIPlusForwardLaucher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int aligned_height,
    const int aligned_width, const float* bottom_rois,
    float* top_data, cudaStream_t stream);

__global__ void ROIPlusBackward(const int nthreads, const float* top_diff,
    const float spatial_scale, const int height, const int width,
    const int channels, const int aligned_height, const int aligned_width,
    float* bottom_diff, const float* bottom_rois, float* offset_x_diff, float* offset_y_diff, const float* feature);

int ROIPlusBackwardLaucher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int aligned_height,
    const int aligned_width, const float* bottom_rois,
    float* bottom_diff,
    float* offset_x_diff,
    float* offset_y_diff,
    const float* feature, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

