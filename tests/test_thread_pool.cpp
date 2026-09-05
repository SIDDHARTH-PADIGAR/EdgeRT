#include "edgert/thread_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>

using edgert::ThreadPool;

TEST(ThreadPoolTest, RunsAllSubmittedTasks) {
    ThreadPool pool(4);
    std::atomic<int> counter{0};
    constexpr int kNumTasks = 100;
    for (int i = 0; i < kNumTasks; ++i) {
        pool.submit([&counter] { counter.fetch_add(1, std::memory_order_relaxed); });
    }
    pool.wait_all();
    EXPECT_EQ(counter.load(), kNumTasks);
}

TEST(ThreadPoolTest, ZeroThreadsThrows) {
    EXPECT_THROW(ThreadPool(0), std::invalid_argument);
}

TEST(ThreadPoolTest, WaitAllBlocksUntilTasksFinish) {
    ThreadPool pool(2);
    std::atomic<bool> task_finished{false};
    pool.submit([&task_finished] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        task_finished.store(true);
    });
    pool.wait_all();
    EXPECT_TRUE(task_finished.load());
}

TEST(ThreadPoolTest, TasksActuallyRunConcurrently) {
    constexpr int kNumTasks = 4;
    ThreadPool pool(kNumTasks);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kNumTasks; ++i) {
        pool.submit([] { std::this_thread::sleep_for(std::chrono::milliseconds(100)); });
    }
    pool.wait_all();
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 300);
}

TEST(ThreadPoolTest, DestructorDrainsQueueBeforeJoining) {
    std::atomic<int> counter{0};
    {
        ThreadPool pool(2);
        for (int i = 0; i < 20; ++i) {
            pool.submit([&counter] { counter.fetch_add(1, std::memory_order_relaxed); });
        }
    }
    EXPECT_EQ(counter.load(), 20);
}