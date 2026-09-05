#include "edgert/thread_pool.h"

#include <stdexcept>
#include <utility>

namespace edgert {

ThreadPool::ThreadPool(std::size_t num_threads) {
    if (num_threads == 0) {
        throw std::invalid_argument("ThreadPool: num_threads must be at least 1");
    }
    workers_.reserve(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    task_available_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::submit(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push(std::move(task));
        ++pending_tasks_;
    }
    task_available_.notify_one();
}

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(mutex_);
    all_done_.wait(lock, [this] { return pending_tasks_ == 0; });
}

void ThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            task_available_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

            if (stop_ && tasks_.empty()) {
                return;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
        }

        task();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            --pending_tasks_;
            if (pending_tasks_ == 0) {
                all_done_.notify_all();
            }
        }
    }
}

}  // namespace edgert