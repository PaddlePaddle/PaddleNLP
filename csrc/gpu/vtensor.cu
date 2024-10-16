#include <paddle/extension.h>
#include <paddle/phi/core/dense_tensor.h>

#include <memory>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

// constexpr const size_t cuda_block_size = 2 * 1024 * 1024;
constexpr const size_t cuda_block_size = 32 * 1024 * 1024;
// constexpr const size_t cuda_block_size = 512 * 1024 * 1024;
constexpr const size_t default_reserve_size = 1024 * 1024 * 1024;


class VirtualAllocationMetadata {
public:
    size_t reserved_size = 0;
    std::weak_ptr<phi::Allocation> holder;
    std::vector<CUmemGenericAllocationHandle> physics_block_handle;
};

static std::unordered_map<CUdeviceptr, VirtualAllocationMetadata> virtual_allocation_metadata;


static size_t compute_real_alloc_size(const phi::DenseTensorMeta& meta) {
    size_t num_elements = 1;
    for (int i = 0; i < meta.dims.size(); i++) {
        assert(meta.dims[i] >= 0);
        num_elements *= meta.dims[i];
    }
    size_t alloc_size = phi::SizeOf(meta.dtype) * num_elements;

    size_t real_alloc_num_block = (alloc_size - 1) / cuda_block_size + 1;
    size_t real_alloc_size = real_alloc_num_block * cuda_block_size;

    return real_alloc_size;
}


static void dealloc_virtual_allocation(phi::Allocation* ptr) {
    auto metadata = virtual_allocation_metadata.at((CUdeviceptr)ptr->ptr());
    // std::cout << "releasing virtual allocation with " 
    //     << metadata.physics_block_handle.size() << " blocks" << std::endl;
    
    // cuMemUnmap does not synchronize with the in-flight kernels, so we need to sync here
    int current_device = -1;
    PD_CHECK(cudaGetDevice(&current_device) == CUDA_SUCCESS, "cudaGetDevice before unmap failed");
    PD_CHECK(cudaSetDevice(ptr->place().GetDeviceId()) == CUDA_SUCCESS,
        "cudaSetDevice before unmap failed");
    PD_CHECK(cudaDeviceSynchronize() == CUDA_SUCCESS, "sync before unmap failed");

    CUresult result = CUDA_SUCCESS;
    result = cuMemUnmap((CUdeviceptr)ptr->ptr(), ptr->size());
    PD_CHECK(result == CUDA_SUCCESS, "cuMemUnmap failed");
    for (int i = 0; i < metadata.physics_block_handle.size(); i++) {
        result = cuMemRelease(metadata.physics_block_handle[i]);
        PD_CHECK(result == CUDA_SUCCESS, "cuMemRelease failed");
    }
    result = cuMemAddressFree((CUdeviceptr)ptr->ptr(), metadata.reserved_size);
    PD_CHECK(result == CUDA_SUCCESS, "cuMemAddressFree failed");

    PD_CHECK(cudaSetDevice(current_device) == CUDA_SUCCESS, "cudaSetDevice after unmap failed");

    virtual_allocation_metadata.erase((CUdeviceptr)ptr->ptr());
}


static phi::DenseTensor* alloc_vtensor(
    const phi::DenseTensorMeta& meta,
    const phi::Place& place
) {
    PD_CHECK(
        place.GetType() == phi::AllocationType::GPU,
        "VTensor only supports CUDA device"
    );

    size_t real_alloc_size = compute_real_alloc_size(meta);

    CUresult result = CUDA_SUCCESS;
    
    size_t reserve_size = default_reserve_size;
    CUdeviceptr alloc_ptr;
    result = cuMemAddressReserve(&alloc_ptr, reserve_size, 0, 0, 0);
    PD_CHECK(result == CUDA_SUCCESS, "cuMemAddressReserve failed");

    CUmemGenericAllocationHandle alloc_handle;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = place.GetDeviceId();
    result = cuMemCreate(&alloc_handle, real_alloc_size, &prop, 0);
    PD_CHECK(result == CUDA_SUCCESS, "cuMemCreate failed");

    result = cuMemMap(alloc_ptr, real_alloc_size, 0, alloc_handle, 0);
    PD_CHECK(result == CUDA_SUCCESS, "cuMemMap failed");

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = place.GetDeviceId();
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    result = cuMemSetAccess(alloc_ptr, real_alloc_size, &accessDesc, 1);
    PD_CHECK(result == CUDA_SUCCESS, "cuMemSetAccess failed");

    auto holder = std::make_shared<phi::Allocation>(
        (void*)alloc_ptr, real_alloc_size, dealloc_virtual_allocation, place
    );

    VirtualAllocationMetadata metadata;
    metadata.physics_block_handle.push_back(alloc_handle);
    metadata.reserved_size = reserve_size;
    metadata.holder = holder;
    virtual_allocation_metadata[alloc_ptr] = metadata;

    return new phi::DenseTensor(holder, meta);
}


static phi::DenseTensor* expand_one_token(const phi::DenseTensor& self) {
    auto new_tensor = new phi::DenseTensor(self);
    auto new_meta = new_tensor->meta();
    new_meta.dims[1] += 1;
    new_tensor->set_meta(new_meta);

    size_t required_size = compute_real_alloc_size(new_tensor->meta());
    if (required_size <= new_tensor->capacity()) {
        return new_tensor;
    } else {
        // std::cout << "expanding virtual allocation" << std::endl;

        auto &metadata = virtual_allocation_metadata.at((CUdeviceptr)new_tensor->data());

        PD_CHECK(required_size <= metadata.reserved_size, "reserved size is not enough");

        size_t real_alloc_size = required_size - new_tensor->capacity();
        CUdeviceptr real_alloc_ptr = (CUdeviceptr)new_tensor->data() + new_tensor->capacity();

        CUresult result = CUDA_SUCCESS;

        CUmemGenericAllocationHandle alloc_handle;
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = new_tensor->place().GetDeviceId();
        result = cuMemCreate(&alloc_handle, real_alloc_size, &prop, 0);
        PD_CHECK(result == CUDA_SUCCESS, "cuMemCreate failed during expanding");

        result = cuMemMap(real_alloc_ptr, real_alloc_size, 0, alloc_handle, 0);
        PD_CHECK(result == CUDA_SUCCESS, "cuMemMap failed");

        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = new_tensor->place().GetDeviceId();
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        result = cuMemSetAccess(real_alloc_ptr, real_alloc_size, &accessDesc, 1);
        PD_CHECK(result == CUDA_SUCCESS, "cuMemSetAccess failed");

        metadata.physics_block_handle.push_back(alloc_handle);

        auto holder = metadata.holder.lock();
        PD_CHECK(holder != nullptr, "holder is already released");
        auto holder_for_swap = new phi::Allocation( // For change the size
            holder->ptr(), required_size,
            holder->deleter(), holder->place()
        );
        phi::swap(*holder, *holder_for_swap);
        operator delete(holder_for_swap); // forget the value, no deconstruction

        PD_CHECK(new_tensor->capacity() == required_size);

        return new_tensor;
    }
}


static int64_t get_max_stride(const phi::DDim& strides) {
    int64_t max_stride = 0;
    for (int i = 0; i < strides.size(); i++) {
        max_stride = std::max(max_stride, strides[i]);
    }
    return max_stride;
}


static phi::DDim compute_transposed_strides(const paddle::Tensor& cache) {
    auto shape = cache.shape();
    return {shape[2] * shape[3], shape[0] * shape[2] * shape[3], shape[3], 1};
}



static bool check_contiguous(const paddle::Tensor& x) {
    auto shape = x.shape();
    int64_t num_dims = shape.size();
    const auto& strides = x.strides();

    int64_t num_elements = 1;
    for (int64_t i = num_dims - 1; i >= 0; i--) {
        if (strides[i] != num_elements) {
            return false;
        }
        num_elements *= shape[i];
    }

    return true;
}


std::vector<paddle::Tensor> VTensorReserveOneToken(
    const paddle::Tensor& cache_transposed,
    const paddle::Tensor& append_state,
    bool transposed_input
) {
    // std::cout << "vtensor_reserve_one_token 1 " << (uintptr_t)cache_transposed.data() << std::endl;

    PD_CHECK(cache_transposed.place().GetType() == phi::AllocationType::GPU);
    PD_CHECK(append_state.place().GetType() == phi::AllocationType::GPU);

    PD_CHECK(cache_transposed.shape().size() == 4);
    PD_CHECK(append_state.shape().size() == 4);
    PD_CHECK(append_state.is_dense_tensor());
    PD_CHECK(check_contiguous(append_state));
    PD_CHECK(cache_transposed.dtype() == append_state.dtype());

    paddle::Tensor cache;

    if (transposed_input) {
        cache = paddle::experimental::transpose(cache_transposed, {1, 0, 2, 3});
    } else {
        cache = cache_transposed;
    }
    
    // std::cout << "vtensor_reserve_one_token 2 " << (uintptr_t)cache.data() << std::endl;

    PD_CHECK(cache.is_dense_tensor());
    
    std::shared_ptr<phi::DenseTensor> vtensor;
    if (virtual_allocation_metadata.find((CUdeviceptr)cache.data()) != virtual_allocation_metadata.end()) {
        // std::cout << "found vtensor" << std::endl;
        
        PD_CHECK(get_max_stride(cache.strides()) == cache.strides()[1]);
        vtensor = std::static_pointer_cast<phi::DenseTensor>(cache.impl());
    }
    else {
        // std::cout << "not a vtensor, allocating a new one" << std::endl;

        // phi::DenseTensorMeta meta(cache.dtype(), cache.dims(), cache.strides());
        phi::DenseTensorMeta meta(cache.dtype(), cache.dims(), compute_transposed_strides(cache));
        paddle::Tensor new_tensor{std::shared_ptr<phi::TensorBase>(alloc_vtensor(meta, cache.place()))};
        paddle::experimental::assign_out_(cache, new_tensor);
        vtensor = std::static_pointer_cast<phi::DenseTensor>(new_tensor.impl());
    }

    paddle::Tensor new_tensor{std::shared_ptr<phi::TensorBase>(expand_one_token(*vtensor))};
    
    // This is slow, just use direct copy instead
    // paddle::experimental::set_value_with_tensor_(new_tensor, append_state, {-1}, {INT32_MAX}, {1}, {1}, {}, {});
    size_t dtype_size = phi::SizeOf(new_tensor.dtype());
    size_t offset = (new_tensor.shape()[1] - 1) * new_tensor.strides()[1] * dtype_size;
    cudaMemcpyAsync(
        (void*)((CUdeviceptr)new_tensor.data() + offset), append_state.data(),
        append_state.numel() * dtype_size,
        cudaMemcpyDeviceToDevice, append_state.stream() // FIXME: Tensor.stream() is deprecated
    );
    return {new_tensor};
}


std::vector<std::vector<int64_t>> VTensorReserveOneTokenInferShape(
    const std::vector<int64_t>& cache_shape,
    const std::vector<int64_t>& append_state,
    bool transposed_input
) {
    auto shape = cache_shape;
    if (transposed_input) {
        std::swap(shape[0], shape[1]);
    }

    if (shape[1] >= 0)
        shape[1] += 1;

    return {shape};
}


std::vector<paddle::DataType> VTensorReserveOneTokenInferDtype(
    const paddle::DataType& cache_dtype,
    const paddle::DataType& append_state
) {
    return {cache_dtype};
}


PD_BUILD_OP(vtensor_reserve_one_token)
    .Inputs({"cache", "append_state"})
    .Outputs({"cache_out"})
    .Attrs({"transposed_input: bool"})
    .SetKernelFn(PD_KERNEL(VTensorReserveOneToken))
    .SetInferShapeFn(PD_INFER_SHAPE(VTensorReserveOneTokenInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(VTensorReserveOneTokenInferDtype));
