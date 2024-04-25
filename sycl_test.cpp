/*
 * source /opt/intel/oneapi/compiler/2024.0/env/vars.sh
 * icpx -o sycl_test sycl_test.cpp -fsycl
*/
#include <chrono>
#include <iostream>
#include <cstring>
#include <thread>
#include <sycl/sycl.hpp>

using namespace std::chrono_literals;
std::vector<sycl::device> g_devices;

void init()
{
    std::cout << "Platform list:" << std::endl;
    for (auto &p : sycl::platform::get_platforms())
    {
        std::cout << "\t" << p.get_info<sycl::info::platform::name>() << "\n";
        auto devices = p.get_devices(sycl::info::device_type::all);
        for (auto &d : devices)
        {
            std::cout << "\t\t" << d.get_info<sycl::info::device::name>() << "\n";
        }
    }

    sycl::platform p;
    auto devices = p.get_devices(sycl::info::device_type::gpu);
    std::cout << "Device list of " << p.get_info<sycl::info::platform::name>() << ":" << std::endl;
    for (auto &d : devices)
    {
        std::cout << "\t" << d.get_info<sycl::info::device::name>() << "\n";
    }
    g_devices = devices;
    std::cout << std::endl;
}

void test_q1_to_q2(size_t size, size_t iterations)
{
    sycl::queue q1(g_devices[0]);
    std::cout << "Q1 Running on " << q1.get_device().get_info<sycl::info::device::name>() << "\n";
    size_t *usm_shared_ptr_a = sycl::malloc_shared<size_t>(size, q1, {});

    sycl::queue q2(g_devices[0]);
    size_t *usm_shared_ptr_b = sycl::malloc_shared<size_t>(size, q2, {});
    std::cout << "Q2 Running on " << q2.get_device().get_info<sycl::info::device::name>() << "\n";

    size_t N = size;
    size_t it = 0;
    const auto start = std::chrono::high_resolution_clock::now();
    while (it++ <= iterations)
    {
        q1.submit([&](sycl::handler &h)
                  { h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                   { usm_shared_ptr_a[i] = i; }); })
            .wait();

        std::memcpy(usm_shared_ptr_b, usm_shared_ptr_a, size * sizeof(size_t));
        q2.submit([&](sycl::handler &h)
                  { h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                   { usm_shared_ptr_b[i] *= 2; }); })
            .wait();
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "\tQ1 to Q2 transfer " << size * iterations * sizeof(size_t) << " bytes data costs " << elapsed.count() << " ms"
              << ", bandwidth = " << size * iterations * sizeof(size_t) * 1000 / 1024 / 1024 / elapsed.count() << " MB/s" << std::endl;

    sycl::free(usm_shared_ptr_a, q1);
    sycl::free(usm_shared_ptr_b, q2);

    sycl::queue q(g_devices[0]);
    // std::cout << "Q Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    size_t *usm_device_ptr_a = sycl::malloc_device<size_t>(size, q, {});
    size_t *usm_device_ptr_b = sycl::malloc_device<size_t>(size, q, {});
    const auto start1 = std::chrono::high_resolution_clock::now();
    it = 0;
    while (it++ <= iterations)
    {
        q.submit([&](sycl::handler &h)
                 { h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                  { usm_device_ptr_a[i] = i; }); })
            .wait();
        q.submit([&](sycl::handler &h)
                 { h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                  { usm_device_ptr_b[i] *= 2; }); })
            .wait();
    }
    const auto end1 = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
    std::cout << "\tDevice computation " << size * iterations * sizeof(size_t) << " bytes data costs " << elapsed1.count() << " ms" << std::endl;
    std::cout << "\tD1 to D2 data transfer bandwidth " << size * iterations * sizeof(size_t) * 1000 / 1024 / 1024 / (elapsed.count() - elapsed1.count()) << " MB/s" << std::endl;
    std::cout << std::endl;

    sycl::free(usm_device_ptr_a, q1);
    sycl::free(usm_device_ptr_b, q2);
}

void test_device_to_device(size_t size, size_t iterations)
{
    sycl::queue q(g_devices[0]);
    std::cout << "Q Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    size_t *usm_device_ptr_a = sycl::malloc_device<size_t>(size, q, {});
    size_t *usm_device_ptr_b = sycl::malloc_device<size_t>(size, q, {});

    size_t N = size;
    size_t it = 0;
    const auto start = std::chrono::high_resolution_clock::now();
    while (it++ <= iterations)
    {
        q.memcpy(usm_device_ptr_a, usm_device_ptr_a, size * sizeof(size_t)).wait();
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "\tDevice transfer " << size * iterations * sizeof(size_t) << " bytes data costs " << elapsed.count() << " ms"
              << ", bandwidth = " << size * iterations * sizeof(size_t) * 1000 / 1024 / 1024 / elapsed.count() << " MB/s" << std::endl;
    std::cout << std::endl;
    sycl::free(usm_device_ptr_a, q);
    sycl::free(usm_device_ptr_b, q);
}

void test_host_to_device(size_t size, size_t iterations)
{
    sycl::queue q(g_devices[0]);
    std::cout << "Q Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    size_t *usm_device_ptr_a = sycl::malloc_device<size_t>(size, q, {});
    size_t *usm_host_ptr_b = sycl::malloc_host<size_t>(size, q, {});

    size_t N = size;
    size_t it = 0;
    const auto start = std::chrono::high_resolution_clock::now();
    while (it++ <= iterations)
    {
        q.memcpy(usm_device_ptr_a, usm_host_ptr_b, size * sizeof(size_t)).wait();
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "\tHost to device transfer " << size * iterations * sizeof(size_t) << " bytes data costs " << elapsed.count() << " ms"
              << ", bandwidth = " << size * iterations * sizeof(size_t) * 1000 / 1024 / 1024 / elapsed.count() << " MB/s" << std::endl;
    std::cout << std::endl;
    sycl::free(usm_device_ptr_a, q);
    sycl::free(usm_host_ptr_b, q);
}

void test_device_to_host(size_t size, size_t iterations)
{
    sycl::queue q(g_devices[0]);
    std::cout << "Q Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    size_t *usm_device_ptr_a = sycl::malloc_device<size_t>(size, q, {});
    size_t *usm_host_ptr_b = sycl::malloc_host<size_t>(size, q, {});

    size_t N = size;
    size_t it = 0;
    const auto start = std::chrono::high_resolution_clock::now();
    while (it++ <= iterations)
    {
        q.memcpy(usm_host_ptr_b, usm_device_ptr_a, size * sizeof(size_t)).wait();
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "\tDevice to host transfer " << size * iterations * sizeof(size_t) << " bytes data costs " << elapsed.count() << " ms"
              << ", bandwidth = " << size * iterations * sizeof(size_t) * 1000 / 1024 / 1024 / elapsed.count() << " MB/s" << std::endl;
    std::cout << std::endl;
    sycl::free(usm_device_ptr_a, q);
    sycl::free(usm_host_ptr_b, q);
}

void test_host_to_host(size_t size, size_t iterations)
{
    sycl::queue q(g_devices[0]);
    std::cout << "Q Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    size_t *usm_host_ptr_a = sycl::malloc_host<size_t>(size, q, {});
    size_t *usm_host_ptr_b = sycl::malloc_host<size_t>(size, q, {});

    size_t N = size;
    size_t it = 0;
    const auto start = std::chrono::high_resolution_clock::now();
    while (it++ <= iterations)
    {
        q.memcpy(usm_host_ptr_a, usm_host_ptr_a, size * sizeof(size_t)).wait();
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "\tHost transfer " << size * iterations * sizeof(size_t) << " bytes data costs " << elapsed.count() << " ms"
              << ", bandwidth = " << size * iterations * sizeof(size_t) * 1000 / 1024 / 1024 / elapsed.count() << " MB/s" << std::endl;
    std::cout << std::endl;
    sycl::free(usm_host_ptr_a, q);
    sycl::free(usm_host_ptr_b, q);
}

void test_host_to_host_2(size_t size, size_t iterations)
{
    size_t *host_ptr_a = (size_t *)std::malloc(size * sizeof(size_t));
    size_t *host_ptr_b = (size_t *)std::malloc(size * sizeof(size_t));

    for (auto i = 0; i < size; i++)
    {
        host_ptr_a[i] = i;
    }
    std::cout << "Running on host:" << std::endl;
    size_t N = size;
    size_t it = 0;
    const auto start = std::chrono::high_resolution_clock::now();
    while (it++ <= iterations)
    {
        std::memcpy(host_ptr_b, host_ptr_a, size * sizeof(size_t));
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "\tHost transfer2 " << size * iterations * sizeof(size_t) << " bytes data costs " << elapsed.count() << " ms"
              << ", bandwidth = " << size * iterations * sizeof(size_t) * 1000 / 1024 / 1024 / elapsed.count() << " MB/s" << std::endl;
    std::cout << std::endl;
    std::free(host_ptr_a);
    std::free(host_ptr_b);
}
//
//       A
//            ---> A+B
//       B
//
//
void test_device_to_device_concat(size_t w, size_t h, size_t iterations)
{
    sycl::queue q1(g_devices[0]);
    std::cout << "Q1 Running on " << q1.get_device().get_info<sycl::info::device::name>() << "\n";
    size_t size = w * h;
    char *usm_shared_ptr_a = sycl::malloc_shared<char>(size / 2, q1, {});

    sycl::queue q2(g_devices[1]);
    std::cout << "Q2 Running on " << q2.get_device().get_info<sycl::info::device::name>() << "\n";
    char *usm_shared_ptr_b = sycl::malloc_shared<char>(size, q2, {});

    size_t N = size;
    size_t it = 0;
    std::chrono::_V2::system_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = start - start;
    std::chrono::duration<double, std::milli> fisrt_elapsed = elapsed;
    while (it++ <= iterations)
    {
        // Put data into device
        q1.submit([&](sycl::handler &h)
                  { h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                   { usm_shared_ptr_a[i] = 10; }); })
            .wait();

        q2.submit([&](sycl::handler &h)
                  { h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                   { usm_shared_ptr_b[i] = 1; }); })
            .wait();
        // Transfer data from device A to host side
        start = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < h; i++)
        {
            std::memcpy(usm_shared_ptr_b + w * i + w / 2, usm_shared_ptr_a + w / 2 * i, w / 2 * sizeof(char));
        }
        // Transfer data from host side to device B
        q2.submit([&](sycl::handler &h)
                  { h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                   { usm_shared_ptr_b[i] += 1; }); })
            .wait();
        end = std::chrono::high_resolution_clock::now();
        // std::cout << "   " << (end - start).count() << ", ";
        elapsed += end - start;

        // Exclude EU execution time
        start = std::chrono::high_resolution_clock::now();
        q2.submit([&](sycl::handler &h)
                  { h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                   { usm_shared_ptr_b[i]-=1; }); })
            .wait();
        end = std::chrono::high_resolution_clock::now();
        elapsed -= end - start;
        // std::cout << (end - start).count() << std::endl;
        if (fisrt_elapsed.count() == 0)
            fisrt_elapsed = elapsed;
        std::this_thread::sleep_for(10ms);
    }

    bool stop = false;
    for (size_t i = 0; i < h; i++)
    {
        for (size_t j = 0; j < w / 2; j++)
        {
            if (usm_shared_ptr_b[i * w + w / 2 + j] != usm_shared_ptr_a[i * w / 2 + j])
            {
                std::cout << "Failed at " << i * w / 2 + j << ", src = " << (int)usm_shared_ptr_a[i * w / 2 + j] << ", dst = " << (int)usm_shared_ptr_b[i * w + w / 2 + j] << std::endl;
                stop = true;
                break;
            }
        }
        if (stop)
            break;
    }

    std::cout << "\tDevice A -> B for tensor concat: data size = " << size / 2 * sizeof(char) << ", cost: " << elapsed.count() / iterations << " ms, first_loop = " << fisrt_elapsed.count() << " ms, bandwidth = "
              << size / 2 * iterations * sizeof(char) * 1000 / 1024 / 1024 / elapsed.count() << " MB/s" << std::endl;
    std::cout << std::endl;
    sycl::free(usm_shared_ptr_a, q1);
    sycl::free(usm_shared_ptr_b, q2);
}
//
//       A
//            ---> C
//       B
//
// C(i,j) = A(i,j) + B(i,j)
//
void test_device_to_device_element_add(size_t w, size_t h, size_t iterations)
{
    sycl::queue q1(g_devices[0]);
    std::cout << "Q1 Running on " << q1.get_device().get_info<sycl::info::device::name>() << "\n";
    size_t size = w * h;
    char *usm_shared_ptr_a = sycl::malloc_shared<char>(size, q1, {});

    sycl::queue q2(g_devices[1]);
    std::cout << "Q2 Running on " << q2.get_device().get_info<sycl::info::device::name>() << "\n";
    char *usm_shared_ptr_b = sycl::malloc_shared<char>(size, q2, {});
    char *usm_shared_ptr_c = sycl::malloc_shared<char>(size, q2, {});

    size_t N = size;
    size_t it = 0;
    std::chrono::_V2::system_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = start - start;
    std::chrono::duration<double, std::milli> fisrt_elapsed = elapsed;
    while (it++ <= iterations)
    {
        // Put data into device
        q1.submit([&](sycl::handler &h)
                  { h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                   { usm_shared_ptr_a[i] = 1; }); })
            .wait();
        q2.submit([&](sycl::handler &h)
                  { h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                   { usm_shared_ptr_b[i] = 10; }); })
            .wait();
        std::memset(usm_shared_ptr_c, 0x0, size * sizeof(char));
        // Transfer data from device A to host side
        start = std::chrono::high_resolution_clock::now();
        std::memcpy(usm_shared_ptr_c, usm_shared_ptr_a, size * sizeof(char));
        // Transfer data from host side to device B and do element add operation
        q2.submit([&](sycl::handler &h)
                  { h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                   { usm_shared_ptr_c[i] += usm_shared_ptr_b[i]; }); })
            .wait();
        end = std::chrono::high_resolution_clock::now();
        elapsed += end - start;
        std::this_thread::sleep_for(10ms);
        if (fisrt_elapsed.count() == 0)
            fisrt_elapsed = elapsed;
    }
    
    for(auto i = 0; i< size;i++) {
        if(usm_shared_ptr_c[i] != usm_shared_ptr_a[i] + usm_shared_ptr_b[i])
            std::cout << "Failed at " << i << std::endl;
            break;
    }
    std::cout << "\tDevice A -> B for tensor add: data size = " << size * sizeof(char) << ", cost: " << elapsed.count() / iterations << " ms, first_loop = " << fisrt_elapsed.count() << " ms, bandwidth = "
              << size * iterations * sizeof(char) * 1000 / 1024 / 1024 / elapsed.count() << " MB/s" << std::endl;
    std::cout << std::endl;
    sycl::free(usm_shared_ptr_a, q1);
    sycl::free(usm_shared_ptr_b, q2);
    sycl::free(usm_shared_ptr_c, q2);
}

void test_device_to_device_element_add_ND(size_t w, size_t h, size_t iterations)
{
    sycl::queue q1(g_devices[0]);
    std::cout << "Q1 Running on " << q1.get_device().get_info<sycl::info::device::name>() << "\n";
    size_t size = w * h;
    char *usm_shared_ptr_a = sycl::malloc_shared<char>(size, q1, {});

    sycl::queue q2(g_devices[1]);
    std::cout << "Q2 Running on " << q2.get_device().get_info<sycl::info::device::name>() << "\n";
    char *usm_shared_ptr_b = sycl::malloc_shared<char>(size, q2, {});
    char *usm_shared_ptr_c = sycl::malloc_shared<char>(size, q2, {});

    size_t P = size;
    size_t it = 0;
    std::chrono::_V2::system_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = start - start;
    std::chrono::duration<double, std::milli> fisrt_elapsed = elapsed;
    constexpr size_t N = 1024;
    constexpr size_t M = 1024;
    while (it++ <= iterations)
    {
        // Put data into device
        q1.submit([&](sycl::handler &h)
                  { h.parallel_for(sycl::range<1>(P), [=](sycl::id<1> i)
                                   { usm_shared_ptr_a[i] = 1; }); })
            .wait();
        q2.submit([&](sycl::handler &h)
                  { h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                                   { usm_shared_ptr_b[i] = 1; }); })
            .wait();
        // Transfer data from device A to host side
        start = std::chrono::high_resolution_clock::now();
        // Transfer data from host side to device B and do element add operation
        std::memcpy(usm_shared_ptr_c, usm_shared_ptr_a, size * sizeof(char));
        q2.submit([&](sycl::handler &h)
                  { h.parallel_for(sycl::nd_range<1>(N * M, M), [=](sycl::nd_item<1> index) [[intel::reqd_sub_group_size(16)]]
                                   {  size_t grp_id = index.get_group()[0];
                                      size_t loc_id = index.get_local_id();
                                      size_t id = grp_id * M + loc_id;
                                      usm_shared_ptr_c[id] += usm_shared_ptr_b[id]; }); })
            .wait();
        end = std::chrono::high_resolution_clock::now();
        elapsed += end - start;
        std::this_thread::sleep_for(10ms);
        if (fisrt_elapsed.count() == 0)
            fisrt_elapsed = elapsed;
    }

    for(auto i = 0; i< size;i++) {
        if(usm_shared_ptr_c[i] != usm_shared_ptr_a[i] + usm_shared_ptr_b[i])
            std::cout << "Failed at " << i << std::endl;
            break;
    }

    std::cout << "\tDevice A -> B for tensor add(opt): data size = " << size * sizeof(char) << ", cost: " << elapsed.count() / iterations << " ms, first_loop = " << fisrt_elapsed.count() << "ms, bandwidth = "
              << size * iterations * sizeof(char) * 1000 / 1024 / 1024 / elapsed.count() << " MB/s" << std::endl;
    std::cout << std::endl;
    sycl::free(usm_shared_ptr_a, q1);
    sycl::free(usm_shared_ptr_b, q2);
    sycl::free(usm_shared_ptr_c, q2);
}

void test_host_to_device(size_t w, size_t h, size_t iterations)
{
    sycl::queue q(g_devices[0]);
    size_t size = w * h;
    std::cout << "Q1 Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    char *usm_device_ptr_a = sycl::malloc_device<char>(size, q, {});
    char *usm_host_ptr_b = sycl::malloc_host<char>(size, q, {});

    size_t N = size;
    size_t it = 0;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = start;
    std::chrono::duration<double, std::milli> elapsed = start - start;
    std::chrono::duration<double, std::milli> fisrt_elapsed = elapsed;
    while (it++ <= iterations)
    {
        start = std::chrono::high_resolution_clock::now();
        q.memcpy(usm_device_ptr_a, usm_host_ptr_b, size * sizeof(char)).wait();
        end = std::chrono::high_resolution_clock::now();
        elapsed += end - start;
        std::this_thread::sleep_for(10ms);
        if (fisrt_elapsed.count() == 0)
            fisrt_elapsed = elapsed;
    }
    std::cout << "\tHost -> Device transfer(DMA): data size = " << size * sizeof(char) << ", cost: " << elapsed.count() / iterations << " ms, first_loop = " << fisrt_elapsed.count() << " ms, bandwidth = "
              << size * iterations * sizeof(char) * 1000 / 1024 / 1024 / elapsed.count() << " MB/s" << std::endl;
    std::cout << std::endl;
    sycl::free(usm_device_ptr_a, q);
    sycl::free(usm_host_ptr_b, q);
}

void test_device_to_host_to_device(size_t w, size_t h, size_t iterations)
{
    sycl::queue q1(g_devices[0]);
    size_t size = w * h;
    std::cout << "Q1 Running on " << q1.get_device().get_info<sycl::info::device::name>() << "\n";
    sycl::queue q2(g_devices[1]);
    std::cout << "Q2 Running on " << q2.get_device().get_info<sycl::info::device::name>() << "\n";
    char *usm_device_ptr_a = sycl::malloc_device<char>(size, q1, {});
    char *usm_host_ptr_b1 = sycl::malloc_host<char>(size, q1, {});
    char *usm_host_ptr_b2 = sycl::malloc_host<char>(size, q2, {});
    char *usm_device_ptr_c = sycl::malloc_device<char>(size, q2, {});

    size_t N = size;
    q1.submit([&](sycl::handler &h)
              { h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                               { usm_device_ptr_a[i] = 10; }); })
        .wait();

    size_t it = 0;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = start;
    std::chrono::duration<double, std::milli> elapsed = start - start;
    std::chrono::duration<double, std::milli> fisrt_elapsed = elapsed;
    while (it++ <= iterations)
    {
        start = std::chrono::high_resolution_clock::now();
        q1.memcpy(usm_host_ptr_b1, usm_device_ptr_a, size * sizeof(char)).wait();
        std::memcpy(usm_host_ptr_b2, usm_host_ptr_b1, size * sizeof(char));
        q2.memcpy(usm_device_ptr_c, usm_host_ptr_b2, size * sizeof(char)).wait();
        end = std::chrono::high_resolution_clock::now();
        elapsed += end - start;
        std::this_thread::sleep_for(10ms);
        if (fisrt_elapsed.count() == 0)
            fisrt_elapsed = elapsed;
    }

    q1.memcpy(usm_host_ptr_b1, usm_device_ptr_a, size * sizeof(char)).wait();
    q2.memcpy(usm_host_ptr_b2, usm_device_ptr_c, size * sizeof(char)).wait();
    for (auto i = 0; i < size; i++)
    {
        if (usm_host_ptr_b1[i] != usm_host_ptr_b2[i])
        {
            std::cout << "Failed at " << i << std::endl;
            break;
        }
    }
    std::cout << "\tDevice -> Host -> Device transfer(DMA): data size = " << size * sizeof(char) << ", cost: " << elapsed.count() / iterations << " ms, first_loop = " << fisrt_elapsed.count() << " ms, bandwidth = "
              << size * iterations * sizeof(char) * 1000 / 1024 / 1024 / elapsed.count() << " MB/s" << std::endl;
    std::cout << std::endl;
    sycl::free(usm_device_ptr_a, q1);
    sycl::free(usm_host_ptr_b1, q1);
    sycl::free(usm_host_ptr_b2, q2);
    sycl::free(usm_device_ptr_c, q2);
}

void test_host_to_device_concat(size_t w, size_t h, size_t iterations)
{
    sycl::queue q(g_devices[0]);
    size_t size = w * h;
    std::cout << "Q Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    char *usm_device_ptr_a = sycl::malloc_device<char>(size, q, {});
    char *usm_host_ptr_b = sycl::malloc_host<char>(size / 2, q, {});

    size_t N = size;

    q.submit([&](sycl::handler &h)
             { h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
                              { usm_device_ptr_a[i] = 10; }); })
        .wait();
    size_t it = 0;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = start;
    std::chrono::duration<double, std::milli> elapsed = start - start;
    std::chrono::duration<double, std::milli> fisrt_elapsed = elapsed;
    while (it++ <= iterations)
    {
        start = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < h; i++)
        {
            q.memcpy(usm_device_ptr_a + w * i + w / 2, usm_host_ptr_b + w / 2 * i, w / 2 * sizeof(char));
        }
        end = std::chrono::high_resolution_clock::now();
        elapsed += end - start;
        std::this_thread::sleep_for(10ms);
        if (fisrt_elapsed.count() == 0)
            fisrt_elapsed = elapsed;
    }

    std::cout << "\tHost -> Device transfer for concat: data size = " << size/2 * sizeof(char) << ", cost: " << elapsed.count() / iterations << " ms, first_loop = " << fisrt_elapsed.count() << " ms, bandwidth = " << size / 2 * iterations * sizeof(char) * 1000 / 1024 / 1024 / elapsed.count() << " MB/s" << std::endl;
    std::cout << std::endl;

    sycl::free(usm_device_ptr_a, q);
    sycl::free(usm_host_ptr_b, q);
}

int main()
{
    init();
#if 0
    test_q1_to_q2(1024*1024, 1000);
    test_device_to_device(1024*1024, 1000);
    test_host_to_device(1024*1024, 1000);
    test_device_to_host(1024*1024, 1000);
    test_host_to_host(1024*1024, 1000);
    test_host_to_host_2(1024*1024, 1000);
#endif
    test_device_to_device_concat(1024, 1024, 100);
    test_device_to_device_element_add(1024, 1024, 100);
    // test_device_to_device_element_add_ND(1024, 1024, 100);
    test_host_to_device(1024, 1024, 100);
    test_device_to_host_to_device(1024,1024,100);
    test_host_to_device_concat(1024, 1024, 100);
}
