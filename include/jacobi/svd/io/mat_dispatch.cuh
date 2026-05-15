#pragma once

#include "jacobi/svd/io/mat_metadata.cuh"
#include "jacobi/svd/io/matrix.cuh"
#include "jacobi/svd/io/pinned_host_task_buffer.cuh"

#include <cstddef>
#include <filesystem>
#include <fstream>

namespace jacobi::svd::io
{
    /**
     * @brief 单次派发任务；One dispatch task for a single *.mat matrix record.
     * @note 输入区保存原始网络字节序 payload，工作区用于后续解析/计算；Input region stores raw network-order payload, workspace is reserved for later decode/compute.
     */
    struct MatDispatchTask final
    {
        /**
         * @brief 派发序号；Dispatch sequence index.
         */
        std::size_t sequence_index = 0;

        /**
         * @brief 矩阵行数；Matrix row count.
         */
        std::size_t rows = 0;

        /**
         * @brief 矩阵列数；Matrix column count.
         */
        std::size_t columns = 0;

        /**
         * @brief 单块页锁定缓冲；Single pinned block containing input+workspace.
         */
        PinnedHostTaskBuffer buffer;
    };

    /**
     * @brief *.mat 单游标派发读取器；Single-cursor dispatch reader for *.mat.
     * @note 该类为栈对象设计，不使用 pImpl（pointer to implementation）；This class is stack-allocated and does not use pImpl.
     */
    class MatDispatchReader final
    {
    public:
        /**
         * @brief 通过路径构造派发读取器；Construct dispatch reader from file path.
         * @param path 输入路径；Input path.
         */
        explicit MatDispatchReader(const std::filesystem::path &path);

        /**
         * @brief 读取并填充下一条任务；Read and populate next dispatch task.
         * @param task 输出任务（可移动复用）；Output task (movable and reusable).
         * @param workspace_bytes 预留工作区字节数；Reserved workspace byte size.
         * @return 成功读取返回 true，EOF 返回 false；Returns true if one task is read, false on EOF.
         */
        [[nodiscard]] bool read_next(MatDispatchTask &task, std::size_t workspace_bytes = 0);

    private:
        /**
         * @brief 输入文件流；Input file stream.
         */
        std::ifstream input_;

        /**
         * @brief 下一个派发序号；Next dispatch sequence index.
         */
        std::size_t next_sequence_index_ = 0;
    };

    /**
     * @brief 将派发任务解析为矩阵；Decode dispatch task payload into Matrix.
     * @param task 派发任务；Dispatch task.
     * @return 解码后的矩阵；Decoded matrix.
     */
    [[nodiscard]] Matrix decode_dispatch_task_matrix(const MatDispatchTask &task);
} // namespace jacobi::svd::io
