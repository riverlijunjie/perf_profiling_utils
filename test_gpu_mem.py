import pyopencl as cl
import numpy as np
import time

def measure_cpu_gpu_bandwidth():
    # Create a context and queue
    platform = cl.get_platforms()[0]  # Select the first platform
    device = platform.get_devices()[0]  # Select the first GPU device
    context = cl.Context([device])
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    # Define data size
    data_size = 1024 * 1024 * 256  # 256 MB of data
    np_data = np.random.rand(data_size).astype(np.float32)
    # Create buffers
    buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=np_data.nbytes)
    # Read data from GPU
    start_time = time.time()
    event_read = cl.enqueue_copy(queue, np_data, buffer, is_blocking=False)
    event_read.wait()
    read_time = time.time() - start_time
    # Write data to GPU
    start_time = time.time()
    event_write = cl.enqueue_copy(queue, buffer, np_data, is_blocking=False)
    event_write.wait()
    write_time = time.time() - start_time
    # Calculate bandwidth
    write_bandwidth = (np_data.nbytes / (1024 * 1024 * 1024)) / write_time  # GB/s
    read_bandwidth = (np_data.nbytes / (1024 * 1024 * 1024)) / read_time  # GB/s
    transfer_bandwidth = (np_data.nbytes / (1024 * 1024 * 1024)) / (read_time + write_time) # GB/s
    print(f"CPU Read GPU Bandwidth: {read_bandwidth:.2f} GB/s")
    print(f"CPU Write GPU Bandwidth: {write_bandwidth:.2f} GB/s")
    print(f"CPU Transfer GPU Bandwidth: {transfer_bandwidth:.2f} GB/s")

def measure_gpu_memory_bandwidth():
    # Get platform and device
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    print(device)
    # Create context and command queue
    context = cl.Context([device])
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    # Data size
    data_size = 1024 * 1024 * 256  # 256 MB
    # Initialize data
    data_src = np.random.rand(data_size).astype(np.float32)
    # Create buffers
    buffer_src = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data_src)
    buffer_dst = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, data_src.nbytes)
    # Dummy kernel for device to device transfer
    kernel_code = """
    __kernel void dummy_kernel(__global const float *src, __global float *dst) {
        int gid = get_global_id(0);
        dst[gid] = src[gid];
    }
    """
    program = cl.Program(context, kernel_code).build()
    kernel = cl.Kernel(program, "dummy_kernel")
    kernel.set_arg(0, buffer_src)
    kernel.set_arg(1, buffer_dst)
    # Execute kernel and measure time
    event = cl.enqueue_nd_range_kernel(queue, kernel, (data_size,), None)
    event.wait()
    elapsed_time = (event.profile.end - event.profile.start) * 1e-9  # Convert to seconds
    bandwidth = (data_size * np.dtype(np.float32).itemsize * 2) / (elapsed_time * 1e9)  # GB/s, factor of 2 for read and write
    print(f"GPU W/R Bandwidth: {bandwidth:.2f} GB/s")
if __name__ == "__main__":
    measure_cpu_gpu_bandwidth()
    measure_gpu_memory_bandwidth()