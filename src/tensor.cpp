#include "edgert/tensor.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace edgert {

std::size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::Float32:
            return sizeof(float);
    }
    throw std::invalid_argument("dtype_size: unknown DataType");
}

namespace {

std::vector<int64_t> compute_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

int64_t compute_numel(const std::vector<int64_t>& shape) {
    int64_t numel = 1;
    for (int64_t dim : shape) {
        if (dim < 0) {
            throw std::invalid_argument("Tensor: shape dimensions must be non-negative");
        }
        numel *= dim;
    }
    return numel;
}

std::byte* aligned_allocate(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) {
        return nullptr;
    }
    // std::aligned_alloc requires size to be a multiple of alignment.
    std::size_t rounded = ((bytes + alignment - 1) / alignment) * alignment;
    void* ptr = std::aligned_alloc(alignment, rounded);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return static_cast<std::byte*>(ptr);
}

}  // namespace

void Tensor::AlignedDeleter::operator()(std::byte* ptr) const noexcept {
    std::free(ptr);
}

Tensor::Tensor() noexcept = default;

Tensor::Tensor(std::vector<int64_t> shape, DataType dtype)
    : shape_(std::move(shape)),
      strides_(compute_strides(shape_)),
      dtype_(dtype),
      numel_(compute_numel(shape_)),
      byte_size_(static_cast<std::size_t>(numel_) * dtype_size(dtype_)),
      buffer_(aligned_allocate(byte_size_, kAlignment)) {}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_),
      strides_(other.strides_),
      dtype_(other.dtype_),
      numel_(other.numel_),
      byte_size_(other.byte_size_),
      buffer_(aligned_allocate(other.byte_size_, kAlignment)) {
    if (byte_size_ > 0) {
        std::memcpy(buffer_.get(), other.buffer_.get(), byte_size_);
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;
    }
    Tensor tmp(other);
    *this = std::move(tmp);
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      dtype_(other.dtype_),
      numel_(other.numel_),
      byte_size_(other.byte_size_),
      buffer_(std::move(other.buffer_)) {
    other.numel_ = 0;
    other.byte_size_ = 0;
    other.shape_.clear();
    other.strides_.clear();
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    dtype_ = other.dtype_;
    numel_ = other.numel_;
    byte_size_ = other.byte_size_;
    buffer_ = std::move(other.buffer_);
    other.numel_ = 0;
    other.byte_size_ = 0;
    other.shape_.clear();
    other.strides_.clear();
    return *this;
}

float* Tensor::data() noexcept {
    return reinterpret_cast<float*>(buffer_.get());
}

const float* Tensor::data() const noexcept {
    return reinterpret_cast<const float*>(buffer_.get());
}

int64_t Tensor::flat_index(const std::vector<int64_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::out_of_range("Tensor::at: index rank does not match tensor rank");
    }
    int64_t flat = 0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            throw std::out_of_range("Tensor::at: index out of bounds");
        }
        flat += indices[i] * strides_[i];
    }
    return flat;
}

float& Tensor::at(const std::vector<int64_t>& indices) {
    return data()[flat_index(indices)];
}

float Tensor::at(const std::vector<int64_t>& indices) const {
    return data()[flat_index(indices)];
}

void Tensor::fill(float value) noexcept {
    float* ptr = data();
    for (int64_t i = 0; i < numel_; ++i) {
        ptr[i] = value;
    }
}

}  // namespace edgert