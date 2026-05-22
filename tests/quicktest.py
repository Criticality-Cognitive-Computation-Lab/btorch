import warp as wp

# 尝试初始化
try:
    wp.init()
    print("Devices:", wp.get_devices())
except Exception as e:
    print(f"Init error: {e}")

# 检查 CUDA 是否可用
try:
    import ctypes
    ctypes.CDLL("libcuda.so")
    print("libcuda.so loaded successfully")
except:
    print("Cannot load libcuda.so")

# 检查 Warp 的 CUDA 检测
print(wp.get_cuda_devices())