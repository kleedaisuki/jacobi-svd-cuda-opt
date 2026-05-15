#pragma once

#include "src/application/pipeline_detail.cuh"
#include "src/application/result_writer.cuh"

#include <condition_variable>
#include <exception>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>

namespace jacobi::svd::pipeline::detail
{
        /**
         * @brief 基于 future 队列的异步写出阶段；Asynchronous output stage based on a future queue.
         */
        class FutureQueueOutputStage final
        {
        public:
            /**
             * @brief 构造异步写出阶段；Construct asynchronous output stage.
             * @param output_path 输出路径；Output path.
             * @param format 输出格式；Output format.
             * @param queue_capacity 完成队列容量；Completion queue capacity.
             */
            FutureQueueOutputStage(const std::filesystem::path &output_path,
                                   MatrixFileFormat format,
                                   std::size_t queue_capacity)
                : queue_capacity_(std::max<std::size_t>(queue_capacity, 1)),
                  consumer_thread_(&FutureQueueOutputStage::consumer_main, this, output_path, format)
            {
            }

            /**
             * @brief 析构并安全关闭；Destroy and safely close.
             */
            ~FutureQueueOutputStage()
            {
                close_noexcept();
            }

            /**
             * @brief 禁止拷贝构造；Copy construction is disabled.
             */
            FutureQueueOutputStage(const FutureQueueOutputStage &) = delete;

            /**
             * @brief 禁止拷贝赋值；Copy assignment is disabled.
             * @return 当前对象引用；Reference to current object.
             */
            FutureQueueOutputStage &operator=(const FutureQueueOutputStage &) = delete;

            /**
             * @brief 提交已完成数据包；Submit one completed packet.
             * @param packet 已完成数据包；Completed output packet.
             */
            void submit(std::future<OutputPacket> future_packet)
            {
                std::unique_lock<std::mutex> lock(mutex_);
                producer_cv_.wait(lock, [this] {
                    return future_queue_.size() < queue_capacity_ || closed_ || worker_error_ != nullptr;
                });
                rethrow_worker_error_locked();

                if (closed_)
                {
                    throw std::runtime_error("Output stage is closed.");
                }

                future_queue_.push(std::move(future_packet));
                consumer_cv_.notify_one();
            }

            /**
             * @brief 正常关闭；Close stage in success path.
             * @param expected_count 期望数据包数量；Expected packet count.
             */
            void close_success(std::size_t expected_count)
            {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    expected_count_ = expected_count;
                    closed_ = true;
                }
                consumer_cv_.notify_all();
                producer_cv_.notify_all();
                join_and_rethrow();

                if (written_count_ != expected_count_)
                {
                    throw std::runtime_error("Output stage finished with missing packets.");
                }
            }

            /**
             * @brief 异常关闭；Close stage in error path.
             * @param error 异常对象；Exception object.
             */
            void close_error(std::exception_ptr error)
            {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (worker_error_ == nullptr)
                    {
                        worker_error_ = error;
                    }
                    closed_ = true;
                    abort_ = true;
                }
                consumer_cv_.notify_all();
                producer_cv_.notify_all();
                join_and_rethrow();
            }

        private:
            /**
             * @brief 写线程主循环；Writer-thread main loop.
             * @param output_path 输出路径；Output path.
             * @param format 输出格式；Output format.
             */
            void consumer_main(const std::filesystem::path output_path, MatrixFileFormat format)
            {
                try
                {
                    ensure_output_directory(output_path);
                    ResultWriter writer(output_path, format);

                    for (;;)
                    {
                        std::future<OutputPacket> future_packet;
                        bool has_future = false;

                        {
                            std::unique_lock<std::mutex> lock(mutex_);
                            consumer_cv_.wait(lock, [this] {
                                return !future_queue_.empty() || closed_ || abort_;
                            });

                            if (!future_queue_.empty())
                            {
                                future_packet = std::move(future_queue_.front());
                                future_queue_.pop();
                                has_future = true;
                                producer_cv_.notify_one();
                            }
                            else if (closed_ || abort_)
                            {
                                break;
                            }
                        }

                        if (!has_future)
                        {
                            continue;
                        }

                        OutputPacket packet = future_packet.get();
                        writer.write_packet(packet);
                        ++written_count_;
                    }

                    writer.flush();
                }
                catch (...)
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (worker_error_ == nullptr)
                    {
                        worker_error_ = std::current_exception();
                    }
                    closed_ = true;
                    abort_ = true;
                    producer_cv_.notify_all();
                    consumer_cv_.notify_all();
                }
            }

            /**
             * @brief 连接写线程并抛出异常；Join writer thread and rethrow error if any.
             */
            void join_and_rethrow()
            {
                if (consumer_thread_.joinable())
                {
                    consumer_thread_.join();
                }

                if (worker_error_ != nullptr)
                {
                    std::rethrow_exception(worker_error_);
                }
            }

            /**
             * @brief 在锁内抛出写线程异常；Rethrow writer-thread exception while lock is held.
             */
            void rethrow_worker_error_locked() const
            {
                if (worker_error_ != nullptr)
                {
                    std::rethrow_exception(worker_error_);
                }
            }

            /**
             * @brief noexcept 关闭助手；Noexcept close helper.
             */
            void close_noexcept() noexcept
            {
                try
                {
                    close_error(std::make_exception_ptr(std::runtime_error("Output stage closed by destructor.")));
                }
                catch (...)
                {
                }
            }

            /**
             * @brief 完成队列容量；Completion queue capacity.
             */
            std::size_t queue_capacity_ = 1;

            /**
             * @brief 输出 future 队列；Output future queue.
             */
            std::queue<std::future<OutputPacket>> future_queue_;

            /**
             * @brief 期望总包数；Expected packet count.
             */
            std::size_t expected_count_ = 0;

            /**
             * @brief 已写包数；Written packet count.
             */
            std::size_t written_count_ = 0;

            /**
             * @brief 互斥锁；Mutex.
             */
            std::mutex mutex_;

            /**
             * @brief 消费者条件变量；Consumer condition variable.
             */
            std::condition_variable consumer_cv_;

            /**
             * @brief 生产者条件变量；Producer condition variable.
             */
            std::condition_variable producer_cv_;

            /**
             * @brief 关闭标记；Close flag.
             */
            bool closed_ = false;

            /**
             * @brief 中止标记；Abort flag.
             */
            bool abort_ = false;

            /**
             * @brief 写线程异常；Writer-thread exception.
             */
            std::exception_ptr worker_error_;

            /**
             * @brief 消费线程；Consumer thread.
             */
            std::thread consumer_thread_;
        };
} // namespace jacobi::svd::pipeline::detail
