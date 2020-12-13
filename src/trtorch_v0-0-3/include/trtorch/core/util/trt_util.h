#pragma once

#include "ATen/Tensor.h"
#include "ATen/core/List.h"
#include "NvInfer.h"

namespace nvinfer1 {
inline std::ostream& operator<<(std::ostream& stream, const nvinfer1::DataType& dtype) {
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT: return stream << "Float32";
    case nvinfer1::DataType::kHALF: return stream << "Float16";
    case nvinfer1::DataType::kINT8: return stream << "Int8";
    case nvinfer1::DataType::kINT32: return stream << "Int32";
    default: return stream << "Unknown Data Type";
    }
}

inline bool operator==(const nvinfer1::Dims& in1, const nvinfer1::Dims& in2) {
    if (in1.nbDims != in2.nbDims) {
        return false;
    }

    // TODO maybe look to support broadcasting comparisons

    for (int64_t i = 0; i < in1.nbDims; i++) {
        if (in1.d[i] != in2.d[i]) {
            return false;
        }
    }
    return true;
}

inline bool operator!=(const nvinfer1::Dims& in1, const nvinfer1::Dims& in2) {
    return !(in1 == in2);
}

template <typename T>
inline std::ostream& printSequence(std::ostream& stream, const T* begin, int count) {
    stream << "[";
    if (count > 0) {
        std::copy_n(begin, count - 1, std::ostream_iterator<T>(stream, ", "));
        stream << begin[count - 1];
    }
    stream << "]";
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const nvinfer1::Dims& shape) {
    return printSequence(stream, shape.d, shape.nbDims);
}

inline std::ostream& operator<<(std::ostream& stream, const nvinfer1::Permutation& perm) {
    return printSequence(stream, perm.order, nvinfer1::Dims::MAX_DIMS);
}

inline std::ostream& operator<<(std::ostream& stream, const nvinfer1::DeviceType& dtype) {
    switch (dtype) {
    case nvinfer1::DeviceType::kGPU: return stream << "GPU";
    case nvinfer1::DeviceType::kDLA: return stream << "DLA";
    default: return stream << "Unknown Device Type";
    }
}

inline std::ostream& operator<<(std::ostream& stream, const nvinfer1::EngineCapability& cap) {
    switch (cap) {
    case nvinfer1::EngineCapability::kDEFAULT: return stream << "Default";
    case nvinfer1::EngineCapability::kSAFE_GPU: return stream << "Safe GPU";
    case nvinfer1::EngineCapability::kSAFE_DLA: return stream << "Safe DLA";
    default: return stream << "Unknown Engine Capability Setting";
    }
}
}

namespace trtorch {
namespace core {
namespace util {

int64_t volume(const nvinfer1::Dims& d);

nvinfer1::Dims toDimsPad(c10::IntArrayRef l, uint64_t pad_to);
nvinfer1::Dims toDimsPad(c10::List<int64_t> l, uint64_t pad_to);
nvinfer1::Dims unpadDims(const nvinfer1::Dims& d);
nvinfer1::Dims unsqueezeDims(const nvinfer1::Dims& d, int pos);
nvinfer1::Dims toDims(c10::IntArrayRef l);
nvinfer1::Dims toDims(c10::List<int64_t> l);
nvinfer1::DimsHW toDimsHW(c10::List<int64_t> l);
nvinfer1::DimsHW toDimsHW(c10::IntArrayRef l);
std::vector<int64_t> toVec(nvinfer1::Dims d);
std::string toStr(nvinfer1::Dims d);

at::ScalarType toATenDType(nvinfer1::DataType t);
nvinfer1::DataType toTRTDataType(at::ScalarType t);
c10::optional<nvinfer1::DataType>toTRTDataType(caffe2::TypeMeta dtype);

const std::unordered_map<at::ScalarType, nvinfer1::DataType>& get_aten_trt_type_map();

} // namespace util
} // namespace core
} // namespace trtorch
