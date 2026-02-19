# pragma once

#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <future>
#include <functional>
#include <condition_variable>

class ThreadPool {

    public:
        int nb_threads; 

    private:
        bool stop;
        std::mutex mtx;
        std::condition_variable cv;         
        std::vector<std::thread> workers;           // Vector of threads
        std::queue<std::function<void()>> tasks;    // Queue of tasks

    public:
        ThreadPool();
        ~ThreadPool();
        
        // Unique Instance to avoid Overhead
        static ThreadPool& instance();


        template<typename F>
        std::future<void> enqueue(F&& f) {

            // Forward to keep state of variable F either rvalue or lvalue 
            // Packaged_task to avoid the management of a promise
            // Make_shared create a pointer to able to copy (emplace) Packaged_task
            auto task = std::make_shared<std::packaged_task<void()>>(std::forward<F>(f));
            
            // Future to know end of task, might also have been an exception or an output value
            std::future<void> result = task->get_future(); 
            {
                std::lock_guard<std::mutex> lock(mtx);
                tasks.emplace([task]() { (*task)(); });
            }
            cv.notify_one();

            return result;
        }
};