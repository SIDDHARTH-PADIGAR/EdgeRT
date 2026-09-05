#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace edgert {

// A fixed number of worker threads, created once and reused for many
// small tasks, instead of paying thread-creation cost per task.
//
// submit() adds a task to a shared queue and returns immediately. Each
// idle worker thread picks the next task off the queue as soon as it's
// free. wait_all() blocks until every submitted task has actually run.
//
// The destructor lets any tasks still in the queue finish running before
// it joins the worker threads — it does not cancel or drop queued work.
class ThreadPool {
public:
    // Throws std::invalid_argument if num_threads == 0.
    explicit ThreadPool(std::size_t num_threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void submit(std::function<void()> task);
    void wait_all();

    std::size_t num_threads() const noexcept { return workers_.size(); }

private:
    void worker_loop();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    std::mutex mutex_;
    std::condition_variable task_available_;
    std::condition_variable all_done_;

    std::size_t pending_tasks_ = 0;
    bool stop_ = false;
};

}  // namespace edgert