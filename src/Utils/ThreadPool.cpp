#include "ThreadPool.hpp"

ThreadPool::ThreadPool() // Use physical cores instead of logical threads
    : nb_threads(std::thread::hardware_concurrency()/2), stop(false) 
{
    for (int i = 0; i < nb_threads; i++) {
        workers.emplace_back([this]() {         // Create new thread
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->mtx);
                    this->cv.wait(lock, [this]() {
                        return this->stop || !this->tasks.empty();  // Wait until stop or new_task
                    });

                    if (this->stop && this->tasks.empty()) return;  // if stop and no more tasks

                    task = std::move(this->tasks.front());    // Get new task 
                    this->tasks.pop();                        // Unload it from tasks
                }
                task(); // Run task
            }
        });
    }   
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        stop = true;
    }
    cv.notify_all();
    
    for (auto& worker : workers) {
        if (worker.joinable()) worker.join();
    } 
}

ThreadPool& ThreadPool::instance() {
    static ThreadPool pool;
    return pool;
}
