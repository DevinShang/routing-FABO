#pragma once

#include "base/tree.h"

#include <string>
#include <vector>

namespace salt {

class FarSaltBuilder {
public:
    struct Stats {
        std::string base_name;
        int anchor_count = 0;
        int root_covered_count = 0;
        int critical_count = 0;
        DTYPE repair_cost = 0;
        DTYPE base_wl = 0;
        double base_stretch = 1.0;
        DTYPE candidate_wl = 0;
        double candidate_stretch = 1.0;
        int local_regions_tried = 0;
        int local_regions_accepted = 0;
        std::vector<Point> primary_root_seed_points;
        std::vector<Point> owner_pool_points;
        std::vector<Point> owner_anchor_points;
    };

    void RunFull(const Net& net, Tree& tree, double eps);

    const Stats& LastStats() const { return last_stats_; }

private:
    void RunImpl(const Net& net, Tree& tree, double eps);

    Stats last_stats_;
};

}  // namespace salt
