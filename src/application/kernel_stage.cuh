#pragma once

#include "src/application/pipeline_detail.cuh"

#include <utility>

namespace jacobi::svd::pipeline::detail
{
        /**
         * @brief 核函数执行阶段；Kernel execution stage.
         */
        class KernelStage final
        {
        public:
            /**
             * @brief 构造核函数阶段；Construct kernel stage.
             * @param config 核函数配置；Kernel configuration.
             */
            explicit KernelStage(JacobiSvdConfig config)
                : config_(config)
            {
            }

            /**
             * @brief 执行一次 testcase 的 SVD；Execute SVD for one testcase.
             * @param testcase 输入矩阵；Input matrix.
             * @param testcase_index 测试用例索引；Testcase index.
             * @return 输出数据包；Output packet.
             */
            [[nodiscard]] OutputPacket execute(const io::Matrix &testcase, std::size_t testcase_index) const
            {
                validate_testcase_matrix(testcase, testcase_index);

                JacobiSvdResult result = one_sided_jacobi_svd(testcase.values,
                                                              testcase.rows,
                                                              testcase.columns,
                                                              config_);

                OutputPacket packet;
                packet.testcase_index = testcase_index;
                packet.sweeps = result.sweeps;
                packet.u = io::Matrix{
                    .rows = result.rows,
                    .columns = result.columns,
                    .values = std::move(result.u),
                };
                packet.sigma = io::Matrix{
                    .rows = 1,
                    .columns = result.columns,
                    .values = std::move(result.sigma),
                };
                packet.v = io::Matrix{
                    .rows = result.columns,
                    .columns = result.columns,
                    .values = std::move(result.v),
                };
                return packet;
            }

        private:
            /**
             * @brief 核函数配置；Kernel configuration.
             */
            JacobiSvdConfig config_{};
        };
} // namespace jacobi::svd::pipeline::detail
