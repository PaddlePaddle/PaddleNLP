#include <cutlass/complex.h>

namespace cutlass {

/// ENUM class for datatypes
enum class DataType {
    kB1, kU2, kU4, kU8,
    kU16, kU32, kU64, kS2,
    kS4, kS8, kS16, kS32,
    kS64, kF16, kBF16, kF32,
    kTF32, kF64, kCF16, kCBF16,
    kCF32, kCTF32, kCF64, kCS2,
    kCS4, kCS8, kCS16, kCS32, 
    kCS64, kCU2, kCU4, kCU8,
    kCU16, kCU32, kCU64, kInvalid
};

/// ENUM class for LayoutTypes
enum class LayoutType {
    kColumnMajor, kRowMajor,
    kColumnMajorInterleaved2, kRowMajorInterleaved2,
    kColumnMajorInterleaved32, kRowMajorInterleaved32,
    kColumnMajorInterleaved64, kRowMajorInterleaved64,
    kTensorNHWC, kTensorNDHWC, kTensorNCHW, kTensorNGHWC,
    kTensorNC32HW32, kTensorNC64HW64, kTensorC32RSK32,
    kTensorC64RSK64
};

/// ENUM class for opcode class


} // namespace cutlass
