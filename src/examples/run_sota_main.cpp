#include "other_methods/interface.h"
#include "third_party_attribution.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr double kEps = 0.5;

struct Features {
    int degree = 0;
    int bbox_w = 0;
    int bbox_h = 0;
    double ar = std::numeric_limits<double>::infinity();
    double lness = 0.0;
};

struct Metrics {
    int wl = 0;
    int sum_pl = 0;
    int max_pl = 0;
    int sum_md = 0;
    int q_total = 0;
    double pt_norm = std::numeric_limits<double>::quiet_NaN();
    double wt_norm = std::numeric_limits<double>::quiet_NaN();
    double alpha = 1.0;
    double avg_stretch = 1.0;
};

struct Candidate {
    std::string method;
    std::string param;
    Method method_enum = Method::FLUTE;
    Metrics metrics;
    int runtime_us = 0;
    UstSaltExtraStats stats;
};

void PrintUsage(const char* argv0) {
    std::cerr << "Usage:\n"
              << "  " << argv0
              << " -nets <.nets> -out <out.csv> --out-cands <cands.csv> [--limit N]\n";
}

uint64_t CoordKey(int x, int y) {
    const auto ux = static_cast<uint32_t>(x);
    const auto uy = static_cast<uint32_t>(y);
    return (static_cast<uint64_t>(ux) << 32) ^ static_cast<uint64_t>(uy);
}

void DedupAndLexicoRoot(salt::Net& net) {
    std::unordered_set<uint64_t> seen;
    seen.reserve(net.pins.size() * 2);
    std::vector<std::shared_ptr<salt::Pin>> pins;
    pins.reserve(net.pins.size());
    for (const auto& pin : net.pins) {
        const uint64_t key = CoordKey(pin->loc.x, pin->loc.y);
        if (!seen.insert(key).second) continue;
        pins.push_back(pin);
    }
    net.pins.swap(pins);
    for (int i = 0; i < static_cast<int>(net.pins.size()); ++i) {
        net.pins[static_cast<size_t>(i)]->id = i;
    }

    int root = 0;
    for (int i = 1; i < static_cast<int>(net.pins.size()); ++i) {
        const auto& a = net.pins[static_cast<size_t>(i)]->loc;
        const auto& b = net.pins[static_cast<size_t>(root)]->loc;
        if (a.x < b.x || (a.x == b.x && a.y < b.y)) root = i;
    }
    if (root != 0) std::swap(net.pins[0], net.pins[static_cast<size_t>(root)]);
    for (int i = 0; i < static_cast<int>(net.pins.size()); ++i) {
        net.pins[static_cast<size_t>(i)]->id = i;
    }
}

int MaxEmptyAnchoredArea(std::vector<std::pair<int, int>> pts, int width, int height) {
    if (width <= 0 || height <= 0) return 0;
    std::vector<std::pair<int, int>> inside;
    inside.reserve(pts.size());
    for (const auto& p : pts) {
        if (p.first > 0 && p.second > 0) inside.push_back(p);
    }
    if (inside.empty()) return width * height;

    std::sort(inside.begin(), inside.end());
    int best = 0;
    int min_y = height;
    size_t i = 0;
    while (i < inside.size()) {
        const int x = inside[i].first;
        best = std::max(best, x * min_y);
        while (i < inside.size() && inside[i].first == x) {
            min_y = std::min(min_y, inside[i].second);
            ++i;
        }
    }
    return std::max(best, width * min_y);
}

double ComputeLness(const std::vector<std::pair<int, int>>& pts) {
    if (pts.empty()) return 0.0;
    int min_x = pts[0].first;
    int max_x = pts[0].first;
    int min_y = pts[0].second;
    int max_y = pts[0].second;
    for (const auto& p : pts) {
        min_x = std::min(min_x, p.first);
        max_x = std::max(max_x, p.first);
        min_y = std::min(min_y, p.second);
        max_y = std::max(max_y, p.second);
    }
    const int width = max_x - min_x;
    const int height = max_y - min_y;
    const int area = width * height;
    if (area <= 0) return 0.0;

    int best = 0;
    std::vector<std::pair<int, int>> transformed;
    transformed.reserve(pts.size());
    for (int mode = 0; mode < 4; ++mode) {
        transformed.clear();
        for (const auto& p : pts) {
            const int x = (mode & 1) ? max_x - p.first : p.first - min_x;
            const int y = (mode & 2) ? max_y - p.second : p.second - min_y;
            transformed.emplace_back(x, y);
        }
        best = std::max(best, MaxEmptyAnchoredArea(transformed, width, height));
    }
    return static_cast<double>(best) / static_cast<double>(area);
}

Features ComputeFeatures(const salt::Net& net) {
    Features out;
    out.degree = static_cast<int>(net.pins.size());
    if (net.pins.empty()) return out;
    int min_x = net.pins[0]->loc.x;
    int max_x = net.pins[0]->loc.x;
    int min_y = net.pins[0]->loc.y;
    int max_y = net.pins[0]->loc.y;
    std::vector<std::pair<int, int>> pts;
    pts.reserve(net.pins.size());
    for (const auto& pin : net.pins) {
        min_x = std::min(min_x, pin->loc.x);
        max_x = std::max(max_x, pin->loc.x);
        min_y = std::min(min_y, pin->loc.y);
        max_y = std::max(max_y, pin->loc.y);
        pts.emplace_back(pin->loc.x, pin->loc.y);
    }
    out.bbox_w = max_x - min_x;
    out.bbox_h = max_y - min_y;
    if (out.bbox_w > 0 && out.bbox_h > 0) {
        out.ar = static_cast<double>(std::max(out.bbox_w, out.bbox_h)) /
                 static_cast<double>(std::min(out.bbox_w, out.bbox_h));
    }
    out.lness = ComputeLness(pts);
    return out;
}

Metrics ComputeMetrics(const salt::Net& net, salt::Tree& tree, int wl_ref) {
    Metrics out;
    const int pin_count = static_cast<int>(net.pins.size());
    if (pin_count <= 0) return out;

    tree.UpdateId();
    const auto nodes = tree.ObtainNodes();
    std::vector<std::pair<int, int>> xy(static_cast<size_t>(pin_count));
    std::vector<long long> dist(static_cast<size_t>(pin_count), -1);
    long long wl = 0;

    for (const auto& node : nodes) {
        if (!node) continue;
        if (node->parent) wl += Dist(node->loc, node->parent->loc);
        if (node->pin && node->pin->id >= 0 && node->pin->id < pin_count) {
            xy[static_cast<size_t>(node->pin->id)] = {node->loc.x, node->loc.y};
            DTYPE pl = 0;
            const salt::TreeNode* cur = node.get();
            while (cur && cur->parent) {
                pl += Dist(cur->loc, cur->parent->loc);
                cur = cur->parent.get();
            }
            dist[static_cast<size_t>(node->pin->id)] = pl;
        }
    }

    long long sum_pl = 0;
    long long max_pl = 0;
    long long sum_md = 0;
    double alpha = 1.0;
    double stretch_sum = 0.0;
    int stretch_count = 0;
    const auto root = xy[0];
    for (int pin = 1; pin < pin_count; ++pin) {
        const long long pl = dist[static_cast<size_t>(pin)];
        sum_pl += pl;
        max_pl = std::max(max_pl, pl);
        const int md = std::abs(root.first - xy[static_cast<size_t>(pin)].first) +
                       std::abs(root.second - xy[static_cast<size_t>(pin)].second);
        sum_md += md;
        const double stretch = md <= 0 ? std::numeric_limits<double>::infinity()
                                       : static_cast<double>(pl) / static_cast<double>(md);
        alpha = std::max(alpha, stretch);
        stretch_sum += stretch;
        ++stretch_count;
    }

    out.wl = static_cast<int>(wl);
    out.sum_pl = static_cast<int>(sum_pl);
    out.max_pl = static_cast<int>(max_pl);
    out.sum_md = static_cast<int>(sum_md);
    out.q_total = static_cast<int>(sum_pl - sum_md);
    out.pt_norm = sum_md > 0 ? static_cast<double>(sum_pl) / static_cast<double>(sum_md)
                             : std::numeric_limits<double>::quiet_NaN();
    out.wt_norm = wl_ref > 0 ? static_cast<double>(wl) / static_cast<double>(wl_ref)
                             : std::numeric_limits<double>::quiet_NaN();
    out.alpha = alpha;
    out.avg_stretch = stretch_count > 0 ? stretch_sum / static_cast<double>(stretch_count) : 1.0;
    return out;
}

Candidate RunCandidate(const salt::Net& net, Method method, int wl_ref) {
    Candidate out;
    out.method_enum = method;
    out.param = method == +Method::FLUTE ? "eps=0.0" : "eps=0.5";
    out.method = method == +Method::SALT_R3 ? "SALT_full" : method._to_string();

    salt::Tree tree;
    const auto t0 = std::chrono::high_resolution_clock::now();
    GetATreeEx(net, tree, method, method == +Method::FLUTE ? 0.0 : kEps, true, &out.stats);
    const auto t1 = std::chrono::high_resolution_clock::now();
    out.runtime_us =
        static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    tree.RemovePhyRedundantSteiner();
    tree.RemoveTopoRedundantSteiner();
    tree.RemoveEmptyChildren();
    out.metrics = ComputeMetrics(net, tree, wl_ref);
    return out;
}

void WriteCandidateHeader(std::ostream& os) {
    os << "suite,design,net_id,degree,bbox_w,bbox_h,AR,Lness,method,param,"
       << "WL,WL_ref,WT_norm,sumPL,maxPL,sumMD,PT_norm,Q_total,alpha,avgStretch,"
       << "runtime_ms,runtime_us,is_picked,ust_support,ust_slope,ust_cert,"
       << "ust_retained,ust_backbone,ust_mass,far_base,far_anchor_count,"
       << "far_root_covered_count,far_critical_count,far_repair_cost,far_base_wl,"
       << "far_base_stretch,far_candidate_wl,far_candidate_stretch,"
       << "far_local_regions_tried,far_local_regions_accepted\n";
}

void WriteCandidateRow(std::ostream& os, const std::string& design, const salt::Net& net,
                       const Features& features, int wl_ref, const Candidate& cand) {
    os << "ICCAD15," << design << "," << net.id << "," << features.degree << ","
       << features.bbox_w << "," << features.bbox_h << "," << std::setprecision(16)
       << features.ar << "," << features.lness << "," << cand.method << ","
       << cand.param << "," << cand.metrics.wl << "," << wl_ref << ","
       << cand.metrics.wt_norm << "," << cand.metrics.sum_pl << ","
       << cand.metrics.max_pl << "," << cand.metrics.sum_md << ","
       << cand.metrics.pt_norm << "," << cand.metrics.q_total << ","
       << cand.metrics.alpha << "," << cand.metrics.avg_stretch << ","
       << static_cast<int>((cand.runtime_us + 500) / 1000) << ","
       << cand.runtime_us << ",0,";

    if (cand.stats.has) {
        os << cand.stats.support_name << "," << cand.stats.slope_index << ","
           << cand.stats.certificate << "," << cand.stats.retained << ","
           << cand.stats.backbone << "," << cand.stats.mass;
    } else {
        os << ",,,,,";
    }
    if (cand.stats.far_has) {
        os << "," << cand.stats.far_base << "," << cand.stats.far_anchor_count << ","
           << cand.stats.far_root_covered_count << "," << cand.stats.far_critical_count
           << "," << cand.stats.far_repair_cost << "," << cand.stats.far_base_wl << ","
           << cand.stats.far_base_stretch << "," << cand.stats.far_candidate_wl << ","
           << cand.stats.far_candidate_stretch << ","
           << cand.stats.far_local_regions_tried << ","
           << cand.stats.far_local_regions_accepted;
    } else {
        os << ",,,,,,,,,,,";
    }
    os << "\n";
}

void WriteSummaryHeader(std::ostream& os) {
    os << "suite,design,net_id,degree,bbox_w,bbox_h,AR,Lness,method,param,"
       << "WL,WL_ref,WT_norm,sumPL,maxPL,sumMD,PT_norm,Q_total,alpha,avgStretch,"
       << "runtime_ms,runtime_us,picked_method,picked_param\n";
}

void WriteSummaryRow(std::ostream& os, const std::string& design, const salt::Net& net,
                     const Features& features, int wl_ref, const Candidate& far) {
    os << "ICCAD15," << design << "," << net.id << "," << features.degree << ","
       << features.bbox_w << "," << features.bbox_h << "," << std::setprecision(16)
       << features.ar << "," << features.lness << ",FABO,eps=0.5,"
       << far.metrics.wl << "," << wl_ref << "," << far.metrics.wt_norm << ","
       << far.metrics.sum_pl << "," << far.metrics.max_pl << "," << far.metrics.sum_md
       << "," << far.metrics.pt_norm << "," << far.metrics.q_total << ","
       << far.metrics.alpha << "," << far.metrics.avg_stretch << ","
       << static_cast<int>((far.runtime_us + 500) / 1000) << "," << far.runtime_us
       << "," << far.method << "," << far.param << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    PrintThirdPartyAttribution();

    std::string nets_file;
    std::string out_csv;
    std::string out_cands_csv;
    int limit = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "-nets" && i + 1 < argc) {
            nets_file = argv[++i];
        } else if (arg == "-out" && i + 1 < argc) {
            out_csv = argv[++i];
        } else if (arg == "--out-cands" && i + 1 < argc) {
            out_cands_csv = argv[++i];
        } else if (arg == "--limit" && i + 1 < argc) {
            limit = std::stoi(argv[++i]);
        } else if (arg == "--methods" || arg == "--eps" || arg == "--wt-max") {
            ++i;  // accepted for compatibility; this runner is intentionally fixed.
        } else {
            PrintUsage(argv[0]);
            return 1;
        }
    }

    if (nets_file.empty() || out_csv.empty() || out_cands_csv.empty()) {
        PrintUsage(argv[0]);
        return 1;
    }

    std::ifstream nets(nets_file);
    if (!nets.good()) {
        std::cerr << "Error: cannot open nets file: " << nets_file << "\n";
        return 1;
    }
    std::string tok;
    while (nets >> tok && tok != "NETS") {}
    if (tok != "NETS") {
        std::cerr << "Error: missing NETS section in " << nets_file << "\n";
        return 1;
    }

    std::string design = nets_file;
    const size_t slash = design.find_last_of("/\\");
    if (slash != std::string::npos) design = design.substr(slash + 1);
    if (design.size() >= 5 && design.substr(design.size() - 5) == ".nets") {
        design.resize(design.size() - 5);
    }

    std::ofstream out(out_csv);
    std::ofstream cands(out_cands_csv);
    if (!out.good() || !cands.good()) {
        std::cerr << "Error: cannot open output CSV files\n";
        return 1;
    }
    WriteSummaryHeader(out);
    WriteCandidateHeader(cands);

    int written = 0;
    while (limit <= 0 || written < limit) {
        salt::Net net;
        if (!net.Read(nets)) break;
        DedupAndLexicoRoot(net);
        const Features features = ComputeFeatures(net);

        Candidate flute = RunCandidate(net, Method::FLUTE, 1);
        const int wl_ref = flute.metrics.wl;
        flute = RunCandidate(net, Method::FLUTE, wl_ref);
        Candidate salt_r3 = RunCandidate(net, Method::SALT_R3, wl_ref);
        Candidate far = RunCandidate(net, Method::FAR_SALT_FULL, wl_ref);

        WriteCandidateRow(cands, design, net, features, wl_ref, flute);
        WriteCandidateRow(cands, design, net, features, wl_ref, salt_r3);
        WriteCandidateRow(cands, design, net, features, wl_ref, far);
        WriteSummaryRow(out, design, net, features, wl_ref, far);

        ++written;
    }

    std::cerr << "Done. nets_written=" << written << "\n";
    return 0;
}
