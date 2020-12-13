#include "object_detector.h"

#include <dirent.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>


namespace ssd300 {

bool ObjectDetector::LoadClassNames(const std::string& class_name_filename) {
  std::ifstream class_name_ifs(class_name_filename);
  if (class_name_ifs.is_open()) {
    std::string class_name;
    while (std::getline(class_name_ifs, class_name)) {
      class_names_.emplace_back(class_name);
    }
    class_name_ifs.close();
  } else {
    std::cerr << "[ObjectDetector::LoadClassNames()] Error: Could not open "
              << class_name_filename << "\n";
    return false;
  }
  
  if (class_names_.size() == 0) {
    std::cerr << "[ObjectDetector::LoadClassNames()] Error: labe names are empty \n";
    return false;
  }

  return true;
}


bool ObjectDetector::LoadInputImagePaths(const std::string& input_directory) {
  DIR* dir;
  struct dirent* entry;
  if ((dir = opendir(input_directory.c_str())) != NULL) {
    while ((entry = readdir(dir)) != NULL) {
      if (entry->d_name[0] != '.') {
        std::string input_image_filename(entry->d_name);
        std::string input_image_path = input_directory + input_image_filename;
        input_image_paths_.emplace_back(input_image_path);
      }
    }
    closedir(dir);
  } else {
    std::cerr << "[ObjectDetector::LoadInputImages()] Error: Could not open "
              << input_directory << "\n";
    return false;
  }
 
  if (input_image_paths_.size() == 0) {
    std::cerr << "[ObjectDetector::LoadInputImages()] Error: input image filenames are empty \n";
    return false;
  }

  return true;
}


void ObjectDetector::Inference(float confidence_threshold, float iou_threshold) {
  std::cout << "=== Empty inferences to warm up === \n\n";
  for (std::size_t i = 0; i < 3; ++i) {
    cv::Mat tmp_image = cv::Mat::zeros(input_image_size_, input_image_size_, CV_32FC3);
    std::vector<ObjectInfo> tmp_results;
    Detect(tmp_image, 1.0, 1.0, tmp_results);
  }
  std::cout << "=== Warming up is done === \n\n\n";

  for (const auto& input_image_path : input_image_paths_) {
    std::cout << "input_image_path = " << input_image_path << "\n";

    cv::Mat input_image = cv::imread(input_image_path);
    if (input_image.empty()) {
      std::cerr << "[ObjectDetector::Run()] Error: Cloud not open "
                << input_image_path << "\n";
      continue;
    }

    std::vector<ObjectInfo> results;
    Detect(input_image, confidence_threshold, iou_threshold, results);

    SaveResultImage(input_image, results, input_image_path);
  }

  return;
}


void ObjectDetector::Detect(const cv::Mat& input_image,
                            float confidence_threshold, float iou_threshold,
                            std::vector<ObjectInfo>& results) {
  torch::NoGradGuard no_grad_guard;

  std::vector<torch::jit::IValue> inputs;
  PreProcess(input_image, inputs);

  auto start_inference = std::chrono::high_resolution_clock::now();
  auto output = model_.forward(inputs).toTuple()->elements();
  auto end_inference = std::chrono::high_resolution_clock::now();

  auto duration_inference = std::chrono::duration_cast<std::chrono::microseconds>(end_inference - start_inference);
  std::cout << "Inference: " << duration_inference.count() << " [micro second] \n";

  // location_offset_tensor ... {Batch=1, 4, Num of default box=8732}
  // 4 ... 0: delta center x, 1: delta center y, 2: delta width, 3: delta height
  // 8732 ... (38^2)*4[ratios] + (19^2)*6[ratios] + (10^2)*6[ratios] + (5^2)*6[ratios] + (3^2)*4[ratios] + (1^2)*4[ratios]
  at::Tensor location_offset_tensor = output[0].toTensor();

  // class_confidence_tensor ... {Batch=1, 81, Num of default box=8732}
  // 81 ... 0: background confidence, 1~80: class confidence
  // 8732 ... (38^2)*4[ratios] + (19^2)*6[ratios] + (10^2)*6[ratios] + (5^2)*6[ratios] + (3^2)*4[ratios] + (1^2)*4[ratios]
  at::Tensor class_confidence_tensor = output[1].toTensor();

  // results ... {Num of obj, 6}
  // 6 ... 0: top-left x, 1: top-left y, 2: bottom-right x, 3: bottom-right y, 4: class score, 5: class id
  PostProcess(location_offset_tensor, class_confidence_tensor,
              confidence_threshold, iou_threshold,
              results);

  std::cout << "\n";

  return;
}


void ObjectDetector::PreProcess(const cv::Mat& input_image,
                                std::vector<torch::jit::IValue>& inputs) {
  cv::Mat resize_image;
  ResizeToFit300(input_image, resize_image);

  cv::Mat preprocess_image;
  CropCenter(resize_image, preprocess_image);

  // 0 ~ 255 ---> 0.0 ~ 1.0
  cv::cvtColor(preprocess_image, preprocess_image, cv::COLOR_BGR2RGB);
  preprocess_image.convertTo(preprocess_image, CV_32FC3, 1.0 / 255.0);

  // 0.0 ~ 1.0 ---> -1.0 ~ 1.0
  preprocess_image = preprocess_image - cv::Scalar(0.5, 0.5, 0.5);
  preprocess_image = preprocess_image / 0.5;

  // input_tensor ... {Batch=1, Height, Width, Channel=3}
  // --->
  // input_tensor ... {Batch=1, Channel=3, Height, Width}
  at::Tensor input_tensor = torch::from_blob(preprocess_image.data,
                                             {1, input_image_size_, input_image_size_, 3});
  input_tensor = input_tensor.permute({0, 3, 1, 2}).contiguous();

  input_tensor = input_tensor.to(torch::kCUDA);
  input_tensor = input_tensor.to(torch::kHalf);

  inputs.clear();
  inputs.emplace_back(input_tensor);

  return;
}


void ObjectDetector::ResizeToFit300(const cv::Mat& src_image, cv::Mat& dst_image) {
  int resize_width = input_image_size_;
  int resize_height = input_image_size_;

  if (src_image.size().height < src_image.size().width) { 
    // landscape
    float ratio = src_image.size().width / static_cast<float>(src_image.size().height);
    resize_width = ceil(ratio * static_cast<float>(input_image_size_));
  } else if (src_image.size().width < src_image.size().height) {
    // portrait
    float ratio = src_image.size().height / static_cast<float>(src_image.size().width);
    resize_height = ceil(ratio * static_cast<float>(input_image_size_));
  }

  cv::resize(src_image, dst_image, cv::Size(resize_width, resize_height));

  return;
}


void ObjectDetector::CropCenter(const cv::Mat& src_image, cv::Mat& dst_image) {
  int center_width = src_image.size().width / 2;
  int center_height = src_image.size().height / 2;
  int half_input_width = input_image_size_ / 2;
  int half_input_height = input_image_size_ / 2;

  cv::Rect center_roi = cv::Rect(cv::Point(center_width - half_input_width, center_height - half_input_height),
                                 cv::Point(center_width + half_input_width, center_height + half_input_height));

  dst_image = src_image(center_roi);

  return;
}


void ObjectDetector::PostProcess(const at::Tensor& location_offset_tensor,
                                 const at::Tensor& class_confidence_tensor,
                                 float confidence_threshold, float iou_threshold,
                                 std::vector<ObjectInfo>& results) {
  int batch_size = location_offset_tensor.size(0);
  if (batch_size != 1) {
    std::cerr << "[ObjectDetector::PostProcess()] Error: Batch size of output tensor is not 1 \n";
    return;
  }


  /*****************************************************************************
   * Prepare default boxes
   ****************************************************************************/

  // location_offset_tensor ... {Batch=1, 4, Num of default box=8732}
  // 4 ... 0: delta center x, 1: delta center y, 2: delta width, 3: delta height
  // 8732 ... (38^2)*4[ratios] + (19^2)*6[ratios] + (10^2)*6[ratios] + (5^2)*6[ratios] + (3^2)*4[ratios] + (1^2)*4[ratios]
  int num_boundbox_info = location_offset_tensor.size(1);  // 4
  int num_defaultbox = location_offset_tensor.size(2);  // 8732

  // defaultbox_xywh_tensor ... {Num of default box=8732, 4}
  // 4 ... 0: delta center x, 1: delta center y, 2: delta width, 3: delta height
  at::Tensor defaultbox_xywh_tensor;
  CreateDefaultboxes(num_defaultbox, num_boundbox_info, defaultbox_xywh_tensor);



  /*****************************************************************************
   * Decode output tensor (location offset) with default boxes
   ****************************************************************************/

  int boundbox_info_dim = -1;  // always in the last dimension
  int class_info_dim = -1;  // always in the last dimension

  // defaultbox_xywh_tensor ... {Num of default box=8732, 4}
  // 4 ... 0: delta center x, 1: delta center y, 2: delta width, 3: delta height
  // --->
  // defaultbox_xy_tensor ... {Num of default box=8732, 2} => similar to [:, 0:2] in Python
  // defaultbox_wh_tensor ... {Num of default box=8732, 2} => similar to [:, 2:4] in Python
  at::Tensor defaultbox_xy_tensor = defaultbox_xywh_tensor.slice(boundbox_info_dim, 0, 2);
  at::Tensor defaultbox_wh_tensor = defaultbox_xywh_tensor.slice(boundbox_info_dim, 2, 4);

  // location_offset_tensor ... {Batch=1, 4, Num of default box=8732}
  // --->
  // offset_xywh_tensor ... {Num of default box=8732, 4}
  at::Tensor offset_xywh_tensor = location_offset_tensor.permute({0, 2, 1}).contiguous()[0];

  // offset_xywh_tensor ... {Num of default box=8732, 4}
  // 4 ... 0: delta center x, 1: delta center y, 2: delta width, 3: delta height
  // --->
  // offset_xy_tensor ... {Num of default box=8732, 2} => similar to [:, 0:2] in Python
  // offset_wh_tensor ... {Num of default box=8732, 2} => similar to [:, 2:4] in Python
  at::Tensor offset_xy_tensor = offset_xywh_tensor.slice(boundbox_info_dim, 0, 2);
  at::Tensor offset_wh_tensor = offset_xywh_tensor.slice(boundbox_info_dim, 2, 4);

  at::Tensor boundbox_xy_tensor = defaultbox_xy_tensor + 0.1 * offset_xy_tensor * defaultbox_wh_tensor;
  at::Tensor boundbox_wh_tensor = defaultbox_wh_tensor * torch::exp(0.2 * offset_wh_tensor);

  // boundbox_xy_tensor ... {Num of default box=8732, 2}
  // boundbox_wh_tensor ... {Num of default box=8732, 2}
  // --->
  // boundbox_xywh_tensor ... {Num of default box=8732, 4}
  at::Tensor boundbox_xywh_tensor = torch::cat({boundbox_xy_tensor, boundbox_wh_tensor}, boundbox_info_dim);

  // boundbox_xywh_tensor ... {Num of default box=8732, 4}
  // 4 ... 0: x center, 1: y center, 2: width, 3: height
  // --->
  // boundbox_tensor ... {Num of default box=8732, 4}
  // 4 ... 0: top-left x, 1: top-left y, 2: bottom-right x, 3: bottom-right y
  at::Tensor boundbox_tensor;
  XcenterYcenterWidthHeight2TopLeftBottomRight(boundbox_xywh_tensor, boundbox_tensor);



  /*****************************************************************************
   * Extract non-background objects
   ****************************************************************************/

  // class_confidence_tensor ... {Batch=1, 81, Num of default box=8732}
  // 81 ... 0: background confidence, 1~80: class confidence
  // --->
  // class_score_tensor ... {Num of default box=8732, 81}
  at::Tensor class_score_tensor = class_confidence_tensor.permute({0, 2, 1}).contiguous()[0];
  class_score_tensor = torch::softmax(class_score_tensor, class_info_dim);

  // max_class_score_tuple ... (value: {Num of default box=8732}, index: {Num of default box=8732})
  std::tuple<at::Tensor, at::Tensor> max_class_score_tuple = torch::max(class_score_tensor,
                                                                        class_info_dim);

  // max_class_score ... {Num of default box=8732}
  // ---> 
  // max_class_score ... {Num of default box=8732, 1}
  at::Tensor max_class_score = std::get<0>(max_class_score_tuple).to(torch::kHalf);
  max_class_score = max_class_score.unsqueeze(class_info_dim);

  // max_class_id ... {Num of default box=8732}
  // --->
  // max_class_id ... {Num of default box=8732, 1}
  at::Tensor max_class_id = std::get<1>(max_class_score_tuple).to(torch::kHalf);
  max_class_id = max_class_id.unsqueeze(class_info_dim);

  // max_class_id ... {Num of default box=8732, 1}
  // --->
  // nonbackground_mask ... {Num of default box=8732, 1}
  at::Tensor nonbackground_mask = max_class_id.select(class_info_dim, 0);
  nonbackground_mask = nonbackground_mask.gt(__FLT_EPSILON__);  // background class id: 0
  nonbackground_mask = nonbackground_mask.unsqueeze(class_info_dim);

  // boundbox_tensor ... {Num of default box=8732, 4}
  // max_class_score ... {Num of default box=8732, 1}
  // max_class_id ... {Num of default box=8732, 1}
  // --->
  // nonbackground_boundbox_tensor ... {Num of non-background box=? * 4}
  // nonbackground_max_class_score ... {Num of non-background box=? * 1}
  // nonbackground_max_class_id ... {Num of non-background box=? * 1}
  at::Tensor nonbackground_boundbox_tensor = torch::masked_select(boundbox_tensor,
                                                                  nonbackground_mask);
  at::Tensor nonbackground_max_class_score = torch::masked_select(max_class_score,
                                                                  nonbackground_mask);
  at::Tensor nonbackground_max_class_id = torch::masked_select(max_class_id,
                                                               nonbackground_mask);
  
  // nonbackground_boundbox_tensor ... {Num of non-background box=? * 4}
  // nonbackground_max_class_score ... {Num of non-background box=? * 1}
  // nonbackground_max_class_id ... {Num of non-background box=? * 1}
  // --->
  // nonbackground_boundbox_tensor ... {Num of non-background box=?, 4}
  // nonbackground_max_class_score ... {Num of non-background box=?, 1}
  // nonbackground_max_class_id ... {Num of non-background box=?, 1}
  nonbackground_boundbox_tensor = nonbackground_boundbox_tensor.view({-1, boundbox_tensor.size(boundbox_info_dim)});
  nonbackground_max_class_score = nonbackground_max_class_score.view({-1, max_class_score.size(class_info_dim)});
  nonbackground_max_class_id = nonbackground_max_class_id.view({-1, max_class_id.size(class_info_dim)});



  /*****************************************************************************
   * Thresholding the non-background objects by class confidence
   ****************************************************************************/

  int boundbox_class_info_dim = -1;  // always in the last dimension
  int object_confidence_idx = 4;

  // nonbackground_result_tensor ... {Num of non-background box=?, 6}
  // 6 ... 0: top-left x, 1: top-left y, 2: bottom-right x, 3: bottom-right y, 4: max class score, 5: max class id
  at::Tensor nonbackground_result_tensor = torch::cat(
    {nonbackground_boundbox_tensor, nonbackground_max_class_score, nonbackground_max_class_id},
    boundbox_class_info_dim);

  // nonbackground_result_tensor ... {Num of non-background box=?, 6}
  // --->
  // candidate_object_mask ... {Num of non-background box=?, 1}
  at::Tensor candidate_object_mask = nonbackground_result_tensor.select(boundbox_class_info_dim,
                                                                        object_confidence_idx);
  candidate_object_mask = candidate_object_mask.gt(confidence_threshold);
  candidate_object_mask = candidate_object_mask.unsqueeze(boundbox_class_info_dim);

  // nonbackground_result_tensor ... {Num of non-background box=?, 6}
  // 6 ... 0: top-left x, 1: top-left y, 2: bottom-right x, 3: bottom-right y, 4: max class score, 5: max class id
  // --->
  // result_tensor ... {Num of candidate box=?? * 6}
  // 6 ... 0: top-left x, 1: top-left y, 2: bottom-right x, 3: bottom-right y, 4: max class score, 5: max class id
  at::Tensor result_tensor = torch::masked_select(nonbackground_result_tensor,
                                                  candidate_object_mask);

  // result_tensor ... {Num of candidate box=?? * 6}
  // --->
  // result_tensor ... {Num of candidate box=??, 6}
  result_tensor = result_tensor.view({-1, nonbackground_result_tensor.size(boundbox_class_info_dim)});

  // If there is no any candidate objects at all, return
  if (result_tensor.size(0) == 0) {
    std::cerr << "[ObjectDetector::PostProcess()] Error: Tthere is no any candidate objects \n";
    return;
  }



  /*****************************************************************************
   * Non Maximum Suppression
   ****************************************************************************/

  // class_id_tensor ... {Num of candidate box=??, 1} => similar to [:, -1:] in Python
  at::Tensor class_id_tensor = result_tensor.slice(class_info_dim, -1);

  // class_offset_bbox_tensor ... {Num of candidate box=??, 4}
  // 4 ... 0: top-left x, 1: top-left y, 2: bottom-right x, 3: bottom-right y (but offset by +4096 * class id)
  at::Tensor class_offset_bbox_tensor = input_image_size_ * result_tensor.slice(boundbox_info_dim, 0, num_boundbox_info)
                                          + nms_max_bbox_size_ * class_id_tensor;

  // Copies tensor to CPU to access tensor elements efficiently with TensorAccessor
  // https://pytorch.org/cppdocs/notes/tensor_basics.html#efficient-access-to-tensor-elements
  at::Tensor class_offset_bbox_tensor_cpu = class_offset_bbox_tensor.to(torch::kFloat).cpu();
  at::Tensor result_tensor_cpu = result_tensor.to(torch::kFloat).cpu();
  auto class_offset_bbox_tensor_accessor = class_offset_bbox_tensor_cpu.accessor<float, 2>();
  auto result_tensor_accessor = result_tensor_cpu.accessor<float, 2>();

  std::vector<cv::Rect> offset_bboxes;
  std::vector<float> class_scores;
  offset_bboxes.reserve(result_tensor_accessor.size(0));
  class_scores.reserve(result_tensor_accessor.size(0));

  for (int i = 0; i < result_tensor_accessor.size(0); ++i) {
    float class_offset_top_left_x = class_offset_bbox_tensor_accessor[i][0];
    float class_offset_top_left_y = class_offset_bbox_tensor_accessor[i][1];
    float class_offset_bottom_right_x = class_offset_bbox_tensor_accessor[i][2];
    float class_offset_bottom_right_y = class_offset_bbox_tensor_accessor[i][3];

    offset_bboxes.emplace_back(cv::Rect(cv::Point(class_offset_top_left_x, class_offset_top_left_y),
                                        cv::Point(class_offset_bottom_right_x, class_offset_bottom_right_y)));

    class_scores.emplace_back(result_tensor_accessor[i][4]);
  }

  std::vector<int> nms_indecies;
  cv::dnn::NMSBoxes(offset_bboxes, class_scores, confidence_threshold, iou_threshold, nms_indecies);



  /*****************************************************************************
   * Create result data
   ****************************************************************************/

  for (const auto& nms_idx : nms_indecies) {
    float top_left_x = result_tensor_accessor[nms_idx][0] * static_cast<float>(input_image_size_);
    float top_left_y = result_tensor_accessor[nms_idx][1] * static_cast<float>(input_image_size_);
    float bottom_right_x = result_tensor_accessor[nms_idx][2] * static_cast<float>(input_image_size_);
    float bottom_right_y = result_tensor_accessor[nms_idx][3] * static_cast<float>(input_image_size_);

    ObjectInfo object_info;
    object_info.bbox_rect = cv::Rect(cv::Point(top_left_x, top_left_y),
                                     cv::Point(bottom_right_x, bottom_right_y));
    object_info.class_score = result_tensor_accessor[nms_idx][4];
    object_info.class_id = result_tensor_accessor[nms_idx][5];

    results.emplace_back(object_info);
  }

  return;
}


void ObjectDetector::CreateDefaultboxes(int num_defaultbox, int num_boundbox_info,
                                        at::Tensor& defaultbox_xywh_tensor) {
  // These parameters can be found here
  // https://github.com/NVIDIA/DeepLearningExamples/blob/7f4ea447296cdbcfe9d2c310c6c1c1557a51b412/PyTorch/Detection/SSD/src/utils.py#L282

  std::vector<int> feature_sizes = {38, 19, 10, 5, 3, 1};
  std::vector<int> feature_steps = {8, 16, 32, 64, 100, 300};

  std::vector<int> smaller_sizes = {21, 45, 99, 153, 207, 261};
  std::vector<int> bigger_sizes = {45, 99, 153, 207, 261, 315};

  std::vector<int> two_rectangle_ratios = {2};
  std::vector<int> four_rectangle_ratios = {2, 3};
  std::vector<std::vector<int>> additional_rectangle_ratios = {two_rectangle_ratios,
                                                               four_rectangle_ratios,
                                                               four_rectangle_ratios,
                                                               four_rectangle_ratios,
                                                               two_rectangle_ratios,
                                                               two_rectangle_ratios};

  float input_image_size_float = static_cast<float>(input_image_size_);
  std::vector<std::vector<float>> defaultboxes_xywh;

  for (std::size_t idx = 0; idx < feature_sizes.size(); ++idx) {
    std::vector<std::vector<float>> defaultbox_wh_pairs;

    float smaller_square_rate = smaller_sizes[idx] / input_image_size_float;  // 0.0 ~ 1.0
    float bigger_square_rate = sqrtf(smaller_sizes[idx] * bigger_sizes[idx]) / input_image_size_float;  // 0.0 ~ 1.0

    std::vector<float> smaller_square_wh_pair = {smaller_square_rate, smaller_square_rate};
    defaultbox_wh_pairs.emplace_back(smaller_square_wh_pair);

    std::vector<float> bigger_square_wh_pair = {bigger_square_rate, bigger_square_rate};
    defaultbox_wh_pairs.emplace_back(bigger_square_wh_pair);

    for (std::size_t i = 0; i < additional_rectangle_ratios[idx].size(); ++i) {
      float longer_rate = smaller_square_rate * sqrtf(additional_rectangle_ratios[idx][i]);
      float shorter_rate = smaller_square_rate / sqrtf(additional_rectangle_ratios[idx][i]);

      std::vector<float> landscape_square_wh_pair = {longer_rate, shorter_rate};
      defaultbox_wh_pairs.emplace_back(landscape_square_wh_pair);

      std::vector<float> portrait_square_wh_pair = {shorter_rate, longer_rate};
      defaultbox_wh_pairs.emplace_back(portrait_square_wh_pair);
    }

    float feature_size = input_image_size_float / feature_steps[idx];

    for (const std::vector<float>& defaultbox_wh_pair : defaultbox_wh_pairs) {
      for (int v = 0; v < feature_sizes[idx]; ++v) {
        for (int u = 0; u < feature_sizes[idx]; ++u) {
          float x_center_rate = (u + 0.5) / feature_size;  // 0.0 ~ 1.0
          float y_center_rate = (v + 0.5) / feature_size;  // 0.0 ~ 1.0

          float width_rate = defaultbox_wh_pair[0];
          float height_rate = defaultbox_wh_pair[1];

          std::vector<float> defaultbox_xywh = {x_center_rate, y_center_rate,
                                                width_rate, height_rate};
          defaultboxes_xywh.emplace_back(defaultbox_xywh);
        }
      }
    }
  }

  if (static_cast<int>(defaultboxes_xywh.size()) != num_defaultbox) {
    std::cerr << "[ObjectDetector::PostProcess()] Error: defaultboxes_xywh.size() != num_defaultbox \n";
    return;
  }

  // std::vector<std::vector<float>> ---> at::Tensor
  // https://stackoverflow.com/a/63467505/10906719
  auto options = torch::TensorOptions().dtype(at::kFloat);

  // defaultbox_xywh_tensor ... {Num of default box=8732, 4}
  defaultbox_xywh_tensor = torch::zeros({num_defaultbox, num_boundbox_info}, options);
  for (int i = 0; i < num_defaultbox; i++) {
    defaultbox_xywh_tensor.slice(0, i, i+1) = torch::from_blob(defaultboxes_xywh[i].data(),
                                                               {num_boundbox_info},
                                                               options);
  }

  defaultbox_xywh_tensor = defaultbox_xywh_tensor.clamp(0.0, 1.0);

  defaultbox_xywh_tensor = defaultbox_xywh_tensor.to(torch::kCUDA);
  defaultbox_xywh_tensor = defaultbox_xywh_tensor.to(torch::kHalf);

  return;
}


// xywh_bbox_tensor ... {Num of bbox, 4}
// 4 ... 0: x center, 1: y center, 2: width, 3: height
// --->
// tlbr_bbox_tensor ... {Num of bbox, 4}
// 4 ... 0: top-left x, 1: top-left y, 2: bottom-right x, 3: bottom-right y
void ObjectDetector::XcenterYcenterWidthHeight2TopLeftBottomRight(const at::Tensor& xywh_bbox_tensor,
                                                                  at::Tensor& tlbr_bbox_tensor) {
  tlbr_bbox_tensor = torch::zeros_like(xywh_bbox_tensor);

  int bbox_dim = -1;  // the last dimension

  int x_center_idx = 0;
  int y_center_idx = 1;
  int width_idx = 2;
  int height_idx = 3;

  tlbr_bbox_tensor.select(bbox_dim, 0) = xywh_bbox_tensor.select(bbox_dim, x_center_idx)
                                           - xywh_bbox_tensor.select(bbox_dim, width_idx).div(2.0);
  tlbr_bbox_tensor.select(bbox_dim, 1) = xywh_bbox_tensor.select(bbox_dim, y_center_idx)
                                           - xywh_bbox_tensor.select(bbox_dim, height_idx).div(2.0);
  tlbr_bbox_tensor.select(bbox_dim, 2) = xywh_bbox_tensor.select(bbox_dim, x_center_idx)
                                           + xywh_bbox_tensor.select(bbox_dim, width_idx).div(2.0);
  tlbr_bbox_tensor.select(bbox_dim, 3) = xywh_bbox_tensor.select(bbox_dim, y_center_idx)
                                           + xywh_bbox_tensor.select(bbox_dim, height_idx).div(2.0);
  
  return;
}


void ObjectDetector::SaveResultImage(const cv::Mat& input_image,
                                     const std::vector<ObjectInfo>& results,
                                     const std::string& input_image_path) {
  cv::Mat resize_image;
  ResizeToFit300(input_image, resize_image);

  cv::Mat result_image;
  CropCenter(resize_image, result_image);

  for (const auto& object_info : results) {
    // Skips background class (class id: 0)
    if (object_info.class_id == 0) {
      continue;
    }

    // Draws object bounding box
    cv::rectangle(result_image, object_info.bbox_rect, cv::Scalar(0,0,255), 1);

    // Class info text
    std::string class_name = class_names_[object_info.class_id - 1];
    std::stringstream class_score;
    class_score << std::fixed << std::setprecision(2) << object_info.class_score;
    std::string class_info = class_name + " " + class_score.str();

    // Size of class info text
    auto font_face = cv::FONT_HERSHEY_SIMPLEX;
    float font_scale = 0.5;
    int thickness = 1;
    int baseline = 0;
    cv::Size class_info_size = cv::getTextSize(class_info, font_face, font_scale, thickness, &baseline);

    // Draws rectangle of class info text
    int height_offset = 5;  // [px]
    cv::Point class_info_top_left = cv::Point(object_info.bbox_rect.tl().x,
                                              object_info.bbox_rect.tl().y - class_info_size.height - height_offset);
    cv::Point class_info_bottom_right = cv::Point(object_info.bbox_rect.tl().x + class_info_size.width,
                                                  object_info.bbox_rect.tl().y);
    cv::rectangle(result_image, class_info_top_left, class_info_bottom_right, cv::Scalar(0,0,255), -1);

    // Draws class info text
    cv::Point class_info_text_position = cv::Point(object_info.bbox_rect.tl().x,
                                                   object_info.bbox_rect.tl().y - height_offset);
    cv::putText(result_image, class_info, class_info_text_position,
                font_face, font_scale, cv::Scalar(0,0,0), thickness);
  }

  std::size_t last_slash_pos = input_image_path.find_last_of('/');
  std::string input_image_directory = input_image_path.substr(0, last_slash_pos + 1);
  std::string input_image_filename = input_image_path.substr(last_slash_pos + 1);

  std::string result_image_directory = input_image_directory + "../result_image/";
  std::string result_image_filename = "result_" + input_image_filename;

  cv::imwrite(result_image_directory + result_image_filename, result_image);

  return;
}

}  // namespace ssd300