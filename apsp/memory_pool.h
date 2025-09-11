#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <hip/hip_runtime.h>
#include <vector>
#include <mutex>
#include <cstdint>
#include <unordered_map>

// 静态全局内存池管理器
class GlobalMemoryPool {
private:
    // 页锁定主机内存池
    struct HostMemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
        
        HostMemoryBlock(void* p, size_t s) : ptr(p), size(s), in_use(false) {}
    };
    
    // GPU内存池
    struct DeviceMemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
        
        DeviceMemoryBlock(void* p, size_t s) : ptr(p), size(s), in_use(false) {}
    };
    
    std::vector<HostMemoryBlock> host_blocks;
    std::vector<DeviceMemoryBlock> device_blocks;
    std::mutex pool_mutex;
    
    // 全局HIP流 - 避免重复创建
    hipStream_t global_stream;
    bool stream_initialized;
    
    // 单例模式
    static GlobalMemoryPool* instance;
    GlobalMemoryPool() : stream_initialized(false) {}
    
public:
    static GlobalMemoryPool& getInstance() {
        if (!instance) {
            instance = new GlobalMemoryPool();
        }
        return *instance;
    }
    
    // 获取页锁定主机内存
    void* allocateHostMemory(size_t bytes) {
        std::lock_guard<std::mutex> lock(pool_mutex);
        
        // 查找可重用的内存块
        for (auto& block : host_blocks) {
            if (!block.in_use && block.size >= bytes) {
                block.in_use = true;
                return block.ptr;
            }
        }
        
        // 分配新内存块 (分配比请求稍大的内存以便后续重用)
        size_t alloc_size = bytes + (bytes >> 3); // 额外分配12.5%
        void* ptr = nullptr;
        hipError_t err = hipHostMalloc(&ptr, alloc_size);
        if (err != hipSuccess) {
            return nullptr;
        }
        
        host_blocks.emplace_back(ptr, alloc_size);
        host_blocks.back().in_use = true;
        return ptr;
    }
    
    // 释放页锁定主机内存 (标记为可重用，不实际释放)
    void deallocateHostMemory(void* ptr) {
        std::lock_guard<std::mutex> lock(pool_mutex);
        for (auto& block : host_blocks) {
            if (block.ptr == ptr) {
                block.in_use = false;
                return;
            }
        }
    }
    
    // 获取GPU内存
    void* allocateDeviceMemory(size_t bytes) {
        std::lock_guard<std::mutex> lock(pool_mutex);
        
        // 查找可重用的内存块
        for (auto& block : device_blocks) {
            if (!block.in_use && block.size >= bytes) {
                block.in_use = true;
                return block.ptr;
            }
        }
        
        // 分配新内存块
        size_t alloc_size = bytes + (bytes >> 3); // 额外分配12.5%
        void* ptr = nullptr;
        hipError_t err = hipMalloc(&ptr, alloc_size);
        if (err != hipSuccess) {
            return nullptr;
        }
        
        device_blocks.emplace_back(ptr, alloc_size);
        device_blocks.back().in_use = true;
        return ptr;
    }
    
    // 释放GPU内存 (标记为可重用，不实际释放)
    void deallocateDeviceMemory(void* ptr) {
        std::lock_guard<std::mutex> lock(pool_mutex);
        for (auto& block : device_blocks) {
            if (block.ptr == ptr) {
                block.in_use = false;
                return;
            }
        }
    }
    
    // 获取全局HIP流 (只创建一次)
    hipStream_t getStream() {
        std::lock_guard<std::mutex> lock(pool_mutex);
        if (!stream_initialized) {
            hipStreamCreate(&global_stream);
            stream_initialized = true;
        }
        return global_stream;
    }
    
    // 程序退出时清理所有内存 (可选调用)
    void cleanup() {
        std::lock_guard<std::mutex> lock(pool_mutex);
        
        for (auto& block : host_blocks) {
            hipHostFree(block.ptr);
        }
        host_blocks.clear();
        
        for (auto& block : device_blocks) {
            hipFree(block.ptr);
        }
        device_blocks.clear();
        
        if (stream_initialized) {
            hipStreamDestroy(global_stream);
            stream_initialized = false;
        }
    }
    
    // 获取池统计信息
    void getPoolStats(size_t& host_total, size_t& host_used, size_t& device_total, size_t& device_used) {
        std::lock_guard<std::mutex> lock(pool_mutex);
        host_total = host_used = device_total = device_used = 0;
        
        for (const auto& block : host_blocks) {
            host_total += block.size;
            if (block.in_use) host_used += block.size;
        }
        
        for (const auto& block : device_blocks) {
            device_total += block.size;
            if (block.in_use) device_used += block.size;
        }
    }
};

// 静态成员定义
GlobalMemoryPool* GlobalMemoryPool::instance = nullptr;

// 便利的RAII包装器
class HostMemoryRAII {
private:
    void* ptr;
    GlobalMemoryPool& pool;
    
public:
    HostMemoryRAII(size_t bytes) : pool(GlobalMemoryPool::getInstance()) {
        ptr = pool.allocateHostMemory(bytes);
    }
    
    ~HostMemoryRAII() {
        if (ptr) {
            pool.deallocateHostMemory(ptr);
        }
    }
    
    void* get() { return ptr; }
    operator bool() const { return ptr != nullptr; }
    
    // 禁止复制，允许移动
    HostMemoryRAII(const HostMemoryRAII&) = delete;
    HostMemoryRAII& operator=(const HostMemoryRAII&) = delete;
    HostMemoryRAII(HostMemoryRAII&& other) noexcept : ptr(other.ptr), pool(other.pool) {
        other.ptr = nullptr;
    }
};

class DeviceMemoryRAII {
private:
    void* ptr;
    GlobalMemoryPool& pool;
    
public:
    DeviceMemoryRAII(size_t bytes) : pool(GlobalMemoryPool::getInstance()) {
        ptr = pool.allocateDeviceMemory(bytes);
    }
    
    ~DeviceMemoryRAII() {
        if (ptr) {
            pool.deallocateDeviceMemory(ptr);
        }
    }
    
    void* get() { return ptr; }
    operator bool() const { return ptr != nullptr; }
    
    // 禁止复制，允许移动  
    DeviceMemoryRAII(const DeviceMemoryRAII&) = delete;
    DeviceMemoryRAII& operator=(const DeviceMemoryRAII&) = delete;
    DeviceMemoryRAII(DeviceMemoryRAII&& other) noexcept : ptr(other.ptr), pool(other.pool) {
        other.ptr = nullptr;
    }
};

// 栈分配器 - 对小对象使用栈内存
template<size_t Size>
class StackAllocator {
private:
    alignas(64) char buffer[Size];  // 64字节对齐
    size_t offset;
    
public:
    StackAllocator() : offset(0) {}
    
    void* allocate(size_t bytes, size_t alignment = 8) {
        // 对齐处理
        size_t aligned_offset = (offset + alignment - 1) & ~(alignment - 1);
        if (aligned_offset + bytes > Size) {
            return nullptr; // 栈空间不足
        }
        
        void* ptr = buffer + aligned_offset;
        offset = aligned_offset + bytes;
        return ptr;
    }
    
    void reset() {
        offset = 0;
    }
    
    size_t remaining() const {
        return Size - offset;
    }
};

#endif // MEMORY_POOL_H
