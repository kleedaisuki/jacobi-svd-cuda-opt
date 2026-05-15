#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace jacobi::svd::pipeline::detail
{
        /**
         * @brief 全局线程池（Thread Pool）实现；Implementation of global thread pool.
         */
        class GlobalThreadPool final
        {
        public:
            /**
             * @brief 构造线程池；Construct thread pool.
             * @param worker_count 工作线程数；Worker thread count.
             */
            explicit GlobalThreadPool(std::size_t worker_count)
                : worker_count_(std::max<std::size_t>(worker_count, 1))
            {
                workers_.reserve(worker_count_);
                for (std::size_t index = 0; index < worker_count_; ++index)
                {
                    workers_.emplace_back([this] {
                        worker_main();
                    });
                }
            }

            /**
             * @brief 析构线程池；Destroy thread pool.
             */
            ~GlobalThreadPool()
            {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    stopping_ = true;
                }
                consumer_cv_.notify_all();

                for (std::thread &worker : workers_)
                {
                    if (worker.joinable())
                    {
                        worker.join();
                    }
                }
            }

            /**
             * @brief 禁止拷贝构造；Copy constructor is disabled.
             */
            GlobalThreadPool(const GlobalThreadPool &) = delete;

            /**
             * @brief 禁止拷贝赋值；Copy assignment is disabled.
             * @return 当前对象引用；Reference to current object.
             */
            GlobalThreadPool &operator=(const GlobalThreadPool &) = delete;

            /**
             * @brief 提交任务并返回 future；Submit task and return future.
             * @tparam Callable 可调用类型；Callable type.
             * @param callable 任务函数；Task callable.
             * @return 任务 future；Task future.
             */
            template <typename Callable>
            [[nodiscard]] auto submit(Callable &&callable) -> std::future<std::invoke_result_t<std::decay_t<Callable>>>
            {
                using ResultType = std::invoke_result_t<std::decay_t<Callable>>;

                auto packaged_task =
                    std::make_shared<std::packaged_task<ResultType()>>(std::forward<Callable>(callable));
                std::future<ResultType> result = packaged_task->get_future();

                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (stopping_)
                    {
                        throw std::runtime_error("Thread pool is stopping.");
                    }
                    queue_.emplace([packaged_task] {
                        (*packaged_task)();
                    });
                }

                consumer_cv_.notify_one();
                return result;
            }

        private:
            /**
             * @brief 工作线程主循环；Worker-thread main loop.
             */
            void worker_main()
            {
                for (;;)
                {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        consumer_cv_.wait(lock, [this] {
                            return stopping_ || !queue_.empty();
                        });

                        if (stopping_ && queue_.empty())
                        {
                            return;
                        }

                        task = std::move(queue_.front());
                        queue_.pop();
                    }

                    task();
                }
            }

            /**
             * @brief 工作线程数量；Worker thread count.
             */
            std::size_t worker_count_ = 1;

            /**
             * @brief 任务队列；Task queue.
             */
            std::queue<std::function<void()>> queue_;

            /**
             * @brief 互斥锁；Mutex.
             */
            std::mutex mutex_;

            /**
             * @brief 条件变量；Condition variable.
             */
            std::condition_variable consumer_cv_;

            /**
             * @brief 停止标志；Stop flag.
             */
            bool stopping_ = false;

            /**
             * @brief 工作线程集合；Worker thread collection.
             */
            std::vector<std::thread> workers_;
        };

        /**
         * @brief 获取默认线程池大小；Get default thread-pool size.
         * @return 工作线程数；Worker thread count.
         */
        [[nodiscard]] std::size_t default_thread_pool_size()
        {
            const unsigned int hardware = std::thread::hardware_concurrency();
            if (hardware == 0U)
            {
                return 4;
            }
            return static_cast<std::size_t>(std::max(1U, hardware));
        }

        /**
         * @brief 全局惰性线程池访问器；Accessor for global lazy-initialized thread pool.
         * @return 全局线程池引用；Reference to global thread pool.
         */
        [[nodiscard]] GlobalThreadPool &global_thread_pool()
        {
            static GlobalThreadPool pool(default_thread_pool_size());
            return pool;
        }
} // namespace jacobi::svd::pipeline::detail
