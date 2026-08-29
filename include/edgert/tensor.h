#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace edgert {

enum class DataType {
    Float32,
};

// Returns the size in bytes of a single element of the given DataType.
std::size_t dtype_size(DataType dtype);

// Tensor is a dense, row-major, contiguous N-dimensional array.
//
// Ownership model:
//   - A Tensor owns its backing storage (RAII, unique_ptr with a custom
//     deleter around an aligned allocation).
//   - Copying a Tensor performs a deep copy of the underlying buffer.
//   - Moving a Tensor transfers ownership; the moved-from Tensor becomes
//     an empty (zero-element) tensor.
//
// Storage is allocated with 64-byte alignment so that future SIMD kernels
// (AVX2/AVX-512 style loads) can operate on tensor data without unaligned
// access penalties.
class Tensor {
public:
    Tensor() noexcept;
    explicit Tensor(std::vector<int64_t> shape, DataType dtype = DataType::Float32);

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor() = default;

    const std::vector<int64_t>& shape() const noexcept { return shape_; }
    const std::vector<int64_t>& strides() const noexcept { return strides_; }
    int64_t ndim() const noexcept { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const noexcept { return numel_; }
    DataType dtype() const noexcept { return dtype_; }
    std::size_t nbytes() const noexcept { return byte_size_; }

    float* data() noexcept;
    const float* data() const noexcept;

    // Element access by multi-dimensional index. Throws std::out_of_range
    // if the index rank does not match the tensor rank, or if any index
    // is out of bounds for its dimension.
    float& at(const std::vector<int64_t>& indices);
    float at(const std::vector<int64_t>& indices) const;

    // Fills every element with the given value.
    void fill(float value) noexcept;

private:
    static constexpr std::size_t kAlignment = 64;

    struct AlignedDeleter {
        void operator()(std::byte* ptr) const noexcept;
    };

    int64_t flat_index(const std::vector<int64_t>& indices) const;

    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    DataType dtype_ = DataType::Float32;
    int64_t numel_ = 0;
    std::size_t byte_size_ = 0;
    std::unique_ptr<std::byte[], AlignedDeleter> buffer_;
};

}  // namespace edgert