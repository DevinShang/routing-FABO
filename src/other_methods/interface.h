#pragma once

#include "salt/base/tree.h"
#include "salt/utils/enum.h"

#include <string>

BETTER_ENUM(Method, int, FLUTE, SALT_R3, FAR_SALT_FULL);

struct UstSaltExtraStats {
    bool has = false;
    std::string support_name;
    int slope_index = -1;
    double certificate = 0.0;
    int retained = 0;
    int backbone = 0;
    int mass = 0;
    int realized_wl = 0;
    double max_stretch = 1.0;

    bool far_has = false;
    std::string far_base;
    int far_anchor_count = 0;
    int far_root_covered_count = 0;
    int far_critical_count = 0;
    int far_repair_cost = 0;
    int far_base_wl = 0;
    double far_base_stretch = 1.0;
    int far_candidate_wl = 0;
    double far_candidate_stretch = 1.0;
    int far_local_regions_tried = 0;
    int far_local_regions_accepted = 0;
};

void GetATree(const salt::Net& net, salt::Tree& tree, Method type, double eps,
              bool checkTree = true);

void GetATreeEx(const salt::Net& net, salt::Tree& tree, Method type, double eps,
                bool checkTree, UstSaltExtraStats* ust_stats);
