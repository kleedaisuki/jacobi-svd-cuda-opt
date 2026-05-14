#include "jacobi/svd/application/pipeline.hpp"

#include "src/application/kernel_stage.hpp"
#include "src/application/output_stage.hpp"
#include "src/application/text_testcase_source.hpp"
#include "src/application/thread_pool.hpp"

#include "jacobi/svd/io/io.hpp"

#include <atomic>
#include <cstddef>
#include <exception>
#include <future>
#include <stdexcept>
#include <utility>

namespace jacobi::svd::pipeline
{
using namespace detail;
    JacobiSvdPipeline::JacobiSvdPipeline(PipelineConfig config)
        : config_(std::move(config))
    {
    }

    PipelineReport JacobiSvdPipeline::run() const
    {
        if (config_.input_path.empty())
        {
            throw std::invalid_argument("Pipeline input path is empty.");
        }
        if (config_.output_path.empty())
        {
            throw std::invalid_argument("Pipeline output path is empty.");
        }

        const MatrixFileFormat input_format = resolve_file_format(config_.input_format, config_.input_path);
        const MatrixFileFormat output_format = resolve_file_format(config_.output_format, config_.output_path);

        JacobiSvdConfig runtime_kernel_config = config_.kernel_config;
        LayoutTransposeAutoTuneReport tuning_report{};
        if (runtime_kernel_config.layout_transpose_auto_tune &&
            runtime_kernel_config.layout_transpose_mode == LayoutTransposeMode::auto_select)
        {
            tuning_report = auto_tune_layout_transpose_threshold(runtime_kernel_config);
            runtime_kernel_config.layout_transpose_min_columns = tuning_report.recommended_min_columns;
            runtime_kernel_config.layout_transpose_min_elements = tuning_report.recommended_min_elements;
            runtime_kernel_config.layout_transpose_auto_tune = false;
        }

        KernelStage kernel(runtime_kernel_config);
        FutureQueueOutputStage output(config_.output_path, output_format, config_.max_queued_results);
        GlobalThreadPool &pool = global_thread_pool();

        std::atomic<std::size_t> total_sweeps{0};
        std::size_t submitted_count = 0;

        try
        {
            if (input_format == MatrixFileFormat::mat)
            {
                io::MatDispatchReader reader(config_.input_path);
                io::MatDispatchTask dispatch_task;

                while (reader.read_next(dispatch_task))
                {
                    io::MatDispatchTask task_for_worker = std::move(dispatch_task);
                    dispatch_task = io::MatDispatchTask{};

                    const std::size_t testcase_index = task_for_worker.sequence_index;
                    std::future<OutputPacket> packet_future = pool.submit([task = std::move(task_for_worker),
                                                                           testcase_index,
                                                                           &kernel,
                                                                           &total_sweeps]() mutable -> OutputPacket {
                        io::Matrix testcase = io::decode_dispatch_task_matrix(task);
                        OutputPacket packet = kernel.execute(testcase, testcase_index);
                        total_sweeps.fetch_add(static_cast<std::size_t>(packet.sweeps), std::memory_order_relaxed);
                        return packet;
                    });
                    output.submit(std::move(packet_future));

                    submitted_count += 1;
                }
            }
            else
            {
                TextTestcaseSource source(config_.input_path);
                io::Matrix testcase;
                std::size_t testcase_index = 0;

                while (source.read_next(testcase))
                {
                    io::Matrix testcase_for_worker = std::move(testcase);
                    testcase = io::Matrix{};

                    std::future<OutputPacket> packet_future = pool.submit([testcase_data = std::move(testcase_for_worker),
                                                                           testcase_index,
                                                                           &kernel,
                                                                           &total_sweeps]() mutable -> OutputPacket {
                        OutputPacket packet = kernel.execute(testcase_data, testcase_index);
                        total_sweeps.fetch_add(static_cast<std::size_t>(packet.sweeps), std::memory_order_relaxed);
                        return packet;
                    });
                    output.submit(std::move(packet_future));

                    submitted_count += 1;
                    testcase_index += 1;
                }
            }

            output.close_success(submitted_count);

            PipelineReport report;
            report.testcase_count = submitted_count;
            report.emitted_matrix_count = submitted_count * 3;
            report.total_sweeps = total_sweeps.load(std::memory_order_relaxed);
            report.layout_transpose_mode = runtime_kernel_config.layout_transpose_mode;
            report.layout_transpose_min_columns = runtime_kernel_config.layout_transpose_min_columns;
            report.layout_transpose_min_elements = runtime_kernel_config.layout_transpose_min_elements;
            report.layout_transpose_auto_tuned = tuning_report.executed;
            report.layout_transpose_estimated_best_speedup = tuning_report.estimated_best_speedup;
            return report;
        }
        catch (...)
        {
            try
            {
                output.close_error(std::current_exception());
            }
            catch (...)
            {
            }
            throw;
        }
    }

    PipelineReport run_pipeline(const PipelineConfig &config)
    {
        return JacobiSvdPipeline(config).run();
    }
} // namespace jacobi::svd::pipeline
