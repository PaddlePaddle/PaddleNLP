#ifndef CUDA_IPC_UTILS_H
#define CUDA_IPC_UTILS_H

#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>


void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }
}

constexpr const char* SHM_NAME = "/cuda_ipc_shm";

constexpr int MAX_PROCESSES = 8;

struct SharedMemory {
    cudaIpcMemHandle_t buffers_handles[MAX_PROCESSES]; 
    cudaIpcMemHandle_t barrier_in_handles[MAX_PROCESSES];
    cudaIpcMemHandle_t barrier_out_handles[MAX_PROCESSES];
    bool ready_flags[MAX_PROCESSES];
};

SharedMemory* create_shared_memory() {
    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Failed to create shared memory" << std::endl;
        exit(EXIT_FAILURE);
    }
    ftruncate(shm_fd, sizeof(SharedMemory));

    void* shm_ptr = mmap(0, sizeof(SharedMemory), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        std::cerr << "Failed to map shared memory" << std::endl;
        exit(EXIT_FAILURE);
    }

    auto* shm = static_cast<SharedMemory*>(shm_ptr);
    std::memset(shm, 0, sizeof(SharedMemory));
    return shm;
}

SharedMemory* open_shared_memory() {
    int shm_fd = shm_open(SHM_NAME, O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Failed to open shared memory" << std::endl;
        exit(EXIT_FAILURE);
    }

    void* shm_ptr = mmap(0, sizeof(SharedMemory), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        std::cerr << "Failed to map shared memory" << std::endl;
        exit(EXIT_FAILURE);
    }

    return static_cast<SharedMemory*>(shm_ptr);
}

void wait_for_ready_flags(const SharedMemory* shm, int total_processes) {
    while (true) {
        bool all_ready = true;
        for (int i = 0; i < total_processes; ++i) {
            if (!shm->ready_flags[i]) {
                all_ready = false;
                break;
            }
        }
        if (all_ready) break;
    }
    usleep(100000);
    std::cout << "All IPC handles are ready." << std::endl;
}

#endif // CUDA_IPC_UTILS_H
