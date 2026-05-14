#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

namespace jacobi::svd::detail
{
    /**
     * @brief 构建巡回赛无冲突列对调度；Build conflict-free round-robin column-pair schedule.
     * @param columns 列数 n；Number of columns n.
     * @return 每个 round 的列对集合；Column-pair groups per round.
     * @note 使用固定首元素的轮转法，保证同一 round 内列不重复；Uses fixed-head rotation to ensure disjoint pairs in each round.
     */
    [[nodiscard]] std::vector<std::vector<int2>> build_round_robin_schedule(int columns)
    {
        if (columns < 2)
        {
            return {};
        }

        const bool needs_dummy = (columns % 2) != 0;
        const int even_columns = needs_dummy ? (columns + 1) : columns;

        std::vector<int> circle(static_cast<std::size_t>(even_columns), -1);
        for (int index = 0; index < columns; ++index)
        {
            circle[static_cast<std::size_t>(index)] = index;
        }

        std::vector<std::vector<int2>> rounds(static_cast<std::size_t>(even_columns - 1));
        for (int round = 0; round < even_columns - 1; ++round)
        {
            auto &pairs = rounds[static_cast<std::size_t>(round)];
            pairs.reserve(static_cast<std::size_t>(even_columns / 2));

            for (int i = 0; i < even_columns / 2; ++i)
            {
                const int lhs = circle[static_cast<std::size_t>(i)];
                const int rhs = circle[static_cast<std::size_t>(even_columns - 1 - i)];
                if (lhs >= 0 && rhs >= 0)
                {
                    pairs.push_back(make_int2(lhs, rhs));
                }
            }

            const int last = circle[static_cast<std::size_t>(even_columns - 1)];
            for (int i = even_columns - 1; i > 1; --i)
            {
                circle[static_cast<std::size_t>(i)] = circle[static_cast<std::size_t>(i - 1)];
            }
            circle[1] = last;
        }

        return rounds;
    }

    /**
     * @brief 规范化线程数量到合法 CUDA 配置；Normalize thread count to valid CUDA launch size.
     * @param raw_threads 用户配置线程数；User-configured thread count.
     * @return 合法且按 warp 对齐的线程数；Valid warp-aligned thread count.
     */
    [[nodiscard]] int normalize_threads_per_block(int raw_threads)
    {
        const int clamped = std::clamp(raw_threads, 32, 1024);
        const int aligned = ((clamped + 31) / 32) * 32;
        return std::min(aligned, 1024);
    }
} // namespace jacobi::svd::detail
