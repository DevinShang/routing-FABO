#include "interface.h"

#include "salt/base/flute.h"
#include "salt/salt.h"
#include "salt/ust_salt.h"

#include <iostream>

void GetATree(const salt::Net& net, salt::Tree& tree, Method type, double eps,
              bool checkTree) {
    GetATreeEx(net, tree, type, eps, checkTree, nullptr);
}

void GetATreeEx(const salt::Net& net, salt::Tree& tree, Method type, double eps,
                bool, UstSaltExtraStats* ust_stats) {
    if (ust_stats) *ust_stats = UstSaltExtraStats{};
    if (eps < 0) {
        std::cerr << "Error: invalid epsilon value\n";
        return;
    }

    switch (type) {
        case Method::FLUTE: {
            salt::FluteBuilder builder;
            builder.Run(net, tree);
            break;
        }
        case Method::SALT_R3: {
            salt::SaltBuilder builder;
            builder.Run(net, tree, eps, 3);
            break;
        }
        case Method::FAR_SALT_FULL: {
            salt::FarSaltBuilder builder;
            builder.RunFull(net, tree, eps);
            if (ust_stats) {
                const auto& fs = builder.LastStats();
                ust_stats->has = true;
                ust_stats->support_name = "FAR:" + fs.base_name;
                ust_stats->slope_index = fs.anchor_count;
                ust_stats->certificate = static_cast<double>(fs.repair_cost);
                ust_stats->retained = static_cast<int>(fs.base_wl);
                ust_stats->backbone = static_cast<int>(fs.candidate_wl);
                ust_stats->mass = fs.critical_count;
                ust_stats->realized_wl = static_cast<int>(fs.candidate_wl);
                ust_stats->max_stretch = fs.candidate_stretch;
                ust_stats->far_has = true;
                ust_stats->far_base = fs.base_name;
                ust_stats->far_anchor_count = fs.anchor_count;
                ust_stats->far_root_covered_count = fs.root_covered_count;
                ust_stats->far_critical_count = fs.critical_count;
                ust_stats->far_repair_cost = static_cast<int>(fs.repair_cost);
                ust_stats->far_base_wl = static_cast<int>(fs.base_wl);
                ust_stats->far_base_stretch = fs.base_stretch;
                ust_stats->far_candidate_wl = static_cast<int>(fs.candidate_wl);
                ust_stats->far_candidate_stretch = fs.candidate_stretch;
                ust_stats->far_local_regions_tried = fs.local_regions_tried;
                ust_stats->far_local_regions_accepted = fs.local_regions_accepted;
            }
            break;
        }
    }
}
