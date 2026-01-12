# pragma once

#include <thread>
#include <vector>
#include <mutex>
#include <functional>
#include <queue>

class ThreadPool {

    private:
        bool stop;
        std::mutex mtx;
        std::condition_variable cv;         
        std::vector<std::thread> workers;           // Vector of threads
        std::queue<std::function<void()>> tasks;    // Queue of tasks
        int nb_threads; 

    public:
        ThreadPool();
        ~ThreadPool();

        template<typename F>
        void enqueue(F&& f) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                tasks.emplace(std::forward<F>(f));
            }
            cv.notify_one();
        }
};