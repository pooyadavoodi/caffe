#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  // Initialize algorithm arrays
  fwd_algo_       = new cudnnConvolutionFwdAlgo_t[bottom.size()];
  bwd_filter_algo_= new cudnnConvolutionBwdFilterAlgo_t[bottom.size()];
  bwd_data_algo_  = new cudnnConvolutionBwdDataAlgo_t[bottom.size()];

  // initialize size arrays
  workspace_fwd_sizes_ = new size_t[bottom.size()];
  workspace_bwd_filter_sizes_ = new size_t[bottom.size()];
  workspace_bwd_data_sizes_ = new size_t[bottom.size()];

  //Initializing algorithms and workspaces
  //Do not rely on initialized algorithms (Reshape will set algorithms with correct values in the first iteration)
  for (size_t i = 0; i < bottom.size(); ++i) {
    fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
    bwd_filter_algo_[i] = (cudnnConvolutionBwdFilterAlgo_t)0;
    bwd_data_algo_[i] = (cudnnConvolutionBwdDataAlgo_t)0;
    workspace_fwd_sizes_[i] = 0;
    workspace_bwd_data_sizes_[i] = 0;
    workspace_bwd_filter_sizes_[i] = 0;
  }

  // Set the indexing parameters.
  bias_offset_ = (this->num_output_ / this->group_);

  // Create filter descriptor.
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int kernel_h = kernel_shape_data[0];
  const int kernel_w = kernel_shape_data[1];
  cudnn::createFilterDesc<Dtype>(&filter_desc_,
      this->num_output_ / this->group_, this->channels_ / this->group_,
      kernel_h, kernel_w);

  this->weight_offset_ = (this->num_output_ / this->group_) *
      (this->channels_ / this->group_) * kernel_h * kernel_w;
  // Create tensor descriptor(s) for data and corresponding convolution(s).
  for (int i = 0; i < bottom.size(); i++) {
    cudnnTensorDescriptor_t bottom_desc;
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc);
    bottom_descs_.push_back(bottom_desc);
    cudnnTensorDescriptor_t top_desc;
    cudnn::createTensor4dDesc<Dtype>(&top_desc);
    top_descs_.push_back(top_desc);
    cudnnConvolutionDescriptor_t conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&conv_desc);
    conv_descs_.push_back(conv_desc);
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
  }

  handles_setup_ = true;
  forward_iter_ = 0;
  //When true, Reshape asks cuDNN for the best algorithm
  use_algo_seeker_ = true;
}

//TODO: Set algorithms again in case parameters (blob shapes) change.
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(2, this->num_spatial_axes_)
      << "CuDNNConvolution input must have 2 spatial axes "
      << "(e.g., height and width). "
      << "Use 'engine: CAFFE' for general ND convolution.";
  bottom_offset_ = this->bottom_dim_ / this->group_;
  top_offset_ = this->top_dim_ / this->group_;
  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int height_out = top[0]->shape(this->channel_axis_ + 1);
  const int width_out = top[0]->shape(this->channel_axis_ + 2);
  const int* pad_data = this->pad_.cpu_data();
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int* stride_data = this->stride_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];

  //Set cuDNN tensor and convolution descriptors
  for (int i = 0; i < bottom.size(); i++) {
    cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i],
        this->num_,
        this->channels_ / this->group_, height, width,
        this->channels_ * height * width,
        height * width, width, 1);
    cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
        this->num_,
        this->num_output_ / this->group_, height_out, width_out,
        this->num_output_ * this->out_spatial_dim_,
        this->out_spatial_dim_, width_out, 1);
    cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i], bottom_descs_[i],
        filter_desc_, pad_h, pad_w, stride_h, stride_w);
  }

  //Ask cuDNN to find the best algorithm
  if(use_algo_seeker_) {
    size_t workspace_limit_bytes, total_memory;
    GPUMemoryManager::GetInfo(&workspace_limit_bytes, &total_memory);
    //FindEx: A workspace of size workspace_bytes is allocated and given to FindEx
    //Get: workspace_bytes is only used as a workspace limit by Get (no allocation happens before Get or by Get)
    size_t workspace_bytes;
    if(forward_iter_ == 0) {
      //In iteration 0, use a small amount of memory in order to leave most of memory for allocating layer blobs
      workspace_bytes = 8*1024*1024;
    }
    else {
      //Use 90% of available memory. Using all of memory may result in failure of workspace.reserve
      //TODO: Since 90% of memory might be too large, we can allocate exactly how much FindEx needs by taking the maximum workspace among all algorithms (requires an initial call to FindEx with workspace size 0)
      workspace_bytes = workspace_limit_bytes * 0.9;
      //Avoid seeking for an algorithm in subsequent iterations
      use_algo_seeker_ = false;
    }
    switch(this->layer_param_.convolution_param().cudnn_convolution_algo_seeker()) {
      case ConvolutionParameter_CuDNNConvolutionAlgorithmSeeker_Get:
        this->GetConvAlgo(bottom, top, workspace_bytes);
        break;
      case ConvolutionParameter_CuDNNConvolutionAlgorithmSeeker_FindEx:
        this->FindExConvAlgo(bottom, top, workspace_bytes);
        break;
      default:
        LOG(ERROR) << "Wrong value for cudnn_convolution_algo_seeker";
        return;
    }
  }

  //At this point, the algorithms and their workspace are set
  //Still need to query cuDNN for workspace size to check whether the selected algorithms are valid because:
    //1) FindEx may return success while giving no valid algorithm as there may be no algorithm available for given parameters
    //2) Algorithms are set in the first 2 iterations, and if parameters change afterwards, validity of selected algorithms should be checked (TODO: Ideally, we should ask cuDNN again for best algorithm if shape of blobs change)
  for (int i = 0; i < bottom.size(); i++) {
    // forward algorithm 
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(Caffe::cudnn_handle(),
        bottom_descs_[i], filter_desc_, conv_descs_[i], top_descs_[i],
        fwd_algo_[i], &(workspace_fwd_sizes_[i])));
    // backward filter algorithm 
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        Caffe::cudnn_handle(),
        bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
        bwd_filter_algo_[i], &workspace_bwd_filter_sizes_[i]));
    // backward data algorithm 
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        Caffe::cudnn_handle(),
        filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
        bwd_data_algo_[i], &workspace_bwd_data_sizes_[i]) );
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
        1, this->num_output_ / this->group_, 1, 1);
  }
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::GetConvAlgo(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const size_t workspace_bytes) {

  for (int i = 0; i < bottom.size(); i++) {
      // Get forward and backward algorithms
      CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(Caffe::cudnn_handle(),
          bottom_descs_[i], filter_desc_, conv_descs_[i], top_descs_[i],
          CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
          workspace_bytes, &fwd_algo_[i]));
      // Get backward algorithm for filter
      CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
          Caffe::cudnn_handle(),
          bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
          CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
          workspace_bytes, &bwd_filter_algo_[i]));
      // Get backward algorithm for data
      CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
          Caffe::cudnn_handle(),
          filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
          CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
          workspace_bytes, &bwd_data_algo_[i]));
  }
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::FindExConvAlgo(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const size_t workspace_bytes) {

  //Number of algorithms we want to consider
  //Since we only consider one algorithm (the fastest), set this to 1
  int request_algo_count = 1;
  int fwd_algo_count, filter_algo_count, data_algo_count;

  cudnnConvolutionFwdAlgoPerf_t *fwd_results = 
    new cudnnConvolutionFwdAlgoPerf_t[request_algo_count];
  cudnnConvolutionBwdFilterAlgoPerf_t *bwd_filter_results = 
    new cudnnConvolutionBwdFilterAlgoPerf_t[request_algo_count];
  cudnnConvolutionBwdDataAlgoPerf_t *bwd_data_results = 
    new cudnnConvolutionBwdDataAlgoPerf_t[request_algo_count];

  //Allocate a temporary buffer for output of cudnnFindConvolutionBackwardFilterAlgorithmEx in order to avoid contaminating this->blobs_[0]
  Dtype *tmp_weights;
  GPUMemoryManager::allocate((void**)&tmp_weights, sizeof(Dtype)*this->weight_offset_);

  //workspace_bytes is the amount of available memory before allocating tmp_weights. So, size of weights should be subtracted from workspace_bytes to represent the correct amount of available memory.
  if(!workspace.try_reserve(workspace_bytes - sizeof(Dtype)*this->weight_offset_)) {
    workspace.reserve(workspace_bytes - sizeof(Dtype)*this->weight_offset_);
  }

  for (int i = 0; i < bottom.size(); i++) {
    //Find forward algorithm
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(Caffe::cudnn_handle(),
                                                       bottom_descs_[i],
                                                       bottom[i]->gpu_data(),
                                                       filter_desc_,
                                                       this->blobs_[0]->gpu_data(),
                                                       conv_descs_[i],
                                                       top_descs_[i],
                                                       top[i]->mutable_gpu_data(),
                                                       request_algo_count,
                                                       &fwd_algo_count,
                                                       fwd_results,
                                                       workspace.data(),
                                                       workspace_bytes));
    fwd_algo_[i] = fwd_results[0].algo;
    workspace_fwd_sizes_[i] = fwd_results[0].memory;

    //Find backward filter algorithm
    CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithmEx(
                                                   Caffe::cudnn_handle(),
                                                   bottom_descs_[i],
                                                   bottom[i]->gpu_data(),
                                                   top_descs_[i],
                                                   top[i]->gpu_diff(),
                                                   conv_descs_[i],
                                                   filter_desc_,
                                                   tmp_weights,
                                                   request_algo_count,
                                                   &filter_algo_count,
                                                   bwd_filter_results,
                                                   workspace.data(),
                                                   workspace_bytes));
    bwd_filter_algo_[i] = bwd_filter_results[0].algo;
    workspace_bwd_filter_sizes_[i] = bwd_filter_results[0].memory;

    //Find backward data algorithm
    CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithmEx(
                                                    Caffe::cudnn_handle(),
                                                    filter_desc_,
                                                    this->blobs_[0]->gpu_data(),
                                                    top_descs_[i],
                                                    top[i]->gpu_diff(),
                                                    conv_descs_[i],
                                                    bottom_descs_[i],
                                                    bottom[i]->mutable_gpu_diff(),
                                                    request_algo_count,
                                                    &data_algo_count,
                                                    bwd_data_results,
                                                    workspace.data(),
                                                    workspace_bytes));

    bwd_data_algo_[i] = bwd_data_results[0].algo;
    workspace_bwd_data_sizes_[i] = bwd_data_results[0].memory;
  }

  GPUMemoryManager::deallocate(tmp_weights);
  workspace.release();
  delete [] fwd_results;
  delete [] bwd_filter_results;
  delete [] bwd_data_results;
}



template <typename Dtype>
CuDNNConvolutionLayer<Dtype>::~CuDNNConvolutionLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  for (int i = 0; i < bottom_descs_.size(); i++) {
    cudnnDestroyTensorDescriptor(bottom_descs_[i]);
    cudnnDestroyTensorDescriptor(top_descs_[i]);
    cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
  }
  if (this->bias_term_) {
    cudnnDestroyTensorDescriptor(bias_desc_);
  }
  cudnnDestroyFilterDescriptor(filter_desc_);

  delete [] fwd_algo_;
  delete [] bwd_filter_algo_;
  delete [] bwd_data_algo_;
  delete [] workspace_fwd_sizes_;
  delete [] workspace_bwd_data_sizes_;
  delete [] workspace_bwd_filter_sizes_;
}

INSTANTIATE_CLASS(CuDNNConvolutionLayer);

}   // namespace caffe
#endif
