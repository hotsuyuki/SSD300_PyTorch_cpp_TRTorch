#ifndef SSD300PYTORCHCPPTRTORCH_OBJECTDETECTOR_OBJECTDETECTOR_H_
#define SSD300PYTORCHCPPTRTORCH_OBJECTDETECTOR_OBJECTDETECTOR_H_


#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include "trtorch/trtorch.h"


namespace ssd300 {

#define DEBUG_PRINT(var) std::cout << #var << " = " << var << "\n";

struct ObjectInfo {
  cv::Rect bbox_rect;
  float class_score;
  int class_id;
};

class ObjectDetector {
 public:
  ObjectDetector(const std::string& model_filename, bool is_optimize)
      : input_image_size_(300),
        nms_max_bbox_size_(4096) {
    std::string height_prefix = "-H";
    std::size_t height_pos = model_filename.find(height_prefix);
    std::string height_string = model_filename.substr(height_pos + height_prefix.length(), 4);
    height_string.erase(height_string.find_last_not_of("-_") + 1);

    int input_height = std::stoi(height_string);
    if (input_height != input_image_size_) {
      std::cerr << "[ObjectDetector()] Error: (input_height)=" << input_height
                << " doesn't match to (input_image_size_)=" << input_image_size_ << "\n";
      std::exit(EXIT_FAILURE);
    }

    std::string width_prefix = "-W";
    std::size_t width_pos = model_filename.find(width_prefix);
    std::string width_string = model_filename.substr(width_pos + width_prefix.length(), 4);
    width_string.erase(width_string.find_last_not_of("-_") + 1);

    int input_width = std::stoi(width_string);
    if (input_width != input_image_size_) {
      std::cerr << "[ObjectDetector()] Error: (input_width)=" << input_width
                << " doesn't match to (input_image_size_)=" << input_image_size_ << "\n";
      std::exit(EXIT_FAILURE);
    }

    // Deserializes the ScriptModule from a file using torch::jit::load()
    // https://pytorch.org/tutorials/advanced/cpp_export.html#a-minimal-c-application
    try {
      std::cout << "[ObjectDetector()] torch::jit::load( " << model_filename << " ); ... \n";
      model_ = torch::jit::load(model_filename);
      std::cout << "[ObjectDetector()] " << model_filename << " has been loaded \n\n";
    }
    catch (const c10::Error& e) {
      std::cerr << e.what() << "\n";
      std::exit(EXIT_FAILURE);
    }
    catch (...) {
      std::cerr << "[ObjectDetector()] Exception: Could not load " << model_filename << "\n";
      std::exit(EXIT_FAILURE);
    }

    std::cout << "Inference on GPU with CUDA \n\n";
    model_.to(torch::kCUDA);
    model_.eval(); 

    std::string torchscript_string = "torchscript";
    std::size_t torchscript_pos = model_filename.find(torchscript_string);
    bool is_found_torchscript = (torchscript_pos != std::string::npos);

    std::string trtorch_string = "trtorch";
    std::size_t trtorch_pos = model_filename.find(trtorch_string);
    bool is_found_trtorch = (trtorch_pos != std::string::npos);

    if (is_found_torchscript && is_optimize) {
      std::cout << "Optimizing TorchScript model with TRTorch \n\n";
      auto in_tensor = torch::randn({1, 3, input_image_size_, input_image_size_});
      auto in_tensor_sizes = std::vector<trtorch::ExtraInfo::InputRange>({in_tensor.sizes()});

      trtorch::ExtraInfo compile_spec_info(in_tensor_sizes);
      compile_spec_info.op_precision = torch::kHalf;

      // Optimizes TorchScript model with TRTorch in FP16
      // https://nvidia.github.io/TRTorch/v0.0.3/tutorials/getting_started.html#compiling-with-trtorch-in-c
      try {
        std::cout << "[ObjectDetector()] trtorch::CompileGraph(); ... \n";
        model_ = trtorch::CompileGraph(model_, compile_spec_info);
        std::cout << "[ObjectDetector()] trtorch::CompileGraph() is done \n\n";
      }
      catch (...) {
        std::cerr << "[ObjectDetector()] Exception: Could not trtorch::CompileGraph() \n";
        std::exit(EXIT_FAILURE);
      }
    } else if (is_found_torchscript && !is_optimize) {
      std::cout << "Without --optimize option: No TRTorch optimization \n\n";
      model_.to(torch::kHalf);
    } else if (!is_found_torchscript && is_found_trtorch) {
      std::cout << "Already optimized by TRTorch: No TRTorch optimization \n\n";
    } else {
      std::cerr << "Unknown model type: No TRTorch optimization \n\n";
    }
  }

  ~ObjectDetector() {}

  bool LoadClassNames(const std::string& class_name_filename);
  
  bool LoadInputImagePaths(const std::string& input_directory);

  void Inference(float confidence_threshold, float iou_threshold);


 private:
  void Detect(const cv::Mat& input_image,
              float confidence_threshold, float iou_threshold,
              std::vector<ObjectInfo>& results);

  void PreProcess(const cv::Mat& input_image,
                  std::vector<torch::jit::IValue>& inputs);

  void ResizeToFit300(const cv::Mat& src_image, cv::Mat& dst_image);

  void CropCenter(const cv::Mat& src_image, cv::Mat& dst_image);

  void PostProcess(const at::Tensor& location_offset_tensor, 
                   const at::Tensor& class_confidence_tensor,
                   float confidence_threshold, float iou_threshold,
                   std::vector<ObjectInfo>& results);
  
  void CreateDefaultboxes(int num_defaultbox, int num_boundbox_info,
                          at::Tensor& defaultbox_xywh_tensor);

  void XcenterYcenterWidthHeight2TopLeftBottomRight(const at::Tensor& xywh_bbox_tensor,
                                                    at::Tensor& tlbr_bbox_tensor);

  void SaveResultImage(const cv::Mat& input_image,
                       const std::vector<ObjectInfo>& results,
                       const std::string& input_image_path);

  int input_image_size_;
  int nms_max_bbox_size_;
  torch::jit::script::Module model_;
  std::vector<std::string> class_names_;
  std::vector<std::string> input_image_paths_;
};

}  // namespace ssd300


#endif  // SSD300PYTORCHCPPTRTORCH_OBJECTDETECTOR_OBJECTDETECTOR_H_