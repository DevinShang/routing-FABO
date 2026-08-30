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
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct Task {
    long long task_id = -1;
    int eps_idx = -1;
    double eps = 0.0;
    std::string design;
    std::string path;
    std::streamoff offset = 0;
};

struct HeaderParams {
    double unit_resistance = 0.0012675;
    double unit_capacitance = 8e-20;
    double driver_resistance = 25.35;
};

struct Features {
    int input_degree = 0;
    int degree = 0;
    int bbox_w = 0;
    int bbox_h = 0;
};

struct Metrics {
    long long wl = 0;
    long long sum_pl = 0;
    long long max_pl = 0;
    long long sum_md = 0;
    double nor_wl = std::numeric_limits<double>::quiet_NaN();
    double nor_path_length = std::numeric_limits<double>::quiet_NaN();
    double alpha = std::numeric_limits<double>::infinity();
    double avg_stretch = std::numeric_limits<double>::infinity();
    double max_delay = std::numeric_limits<double>::quiet_NaN();
    double avg_delay = std::numeric_limits<double>::quiet_NaN();
    double max_nor_delay = std::numeric_limits<double>::quiet_NaN();
    double avg_nor_delay = std::numeric_limits<double>::quiet_NaN();
    bool valid_tree = false;
    bool pins_fixed = false;
    bool all_pins_covered_once = false;
    bool valid_eps = false;
    bool valid_1p5 = false;
};

struct Candidate {
    Metrics metrics;
    long long runtime_us = 0;
    UstSaltExtraStats stats;
};

class ScopedCoutRedirect {
public:
    explicit ScopedCoutRedirect(std::ostream& dst) : old_(std::cout.rdbuf(dst.rdbuf())) {}
    ~ScopedCoutRedirect() { std::cout.rdbuf(old_); }

private:
    std::streambuf* old_;
};

class NullBuffer : public std::streambuf {
public:
    int overflow(int c) override { return c; }
};

uint64_t CoordKey(int x, int y) {
    const auto ux = static_cast<uint32_t>(x);
    const auto uy = static_cast<uint32_t>(y);
    return (static_cast<uint64_t>(ux) << 32) ^ static_cast<uint64_t>(uy);
}

void DedupPreserveInputSource(salt::Net& net) {
    if (net.pins.empty()) return;
    std::unordered_set<uint64_t> seen;
    seen.reserve(net.pins.size() * 2);
    std::vector<std::shared_ptr<salt::Pin>> pins;
    pins.reserve(net.pins.size());

    const uint64_t source_key = CoordKey(net.pins[0]->loc.x, net.pins[0]->loc.y);
    seen.insert(source_key);
    pins.push_back(net.pins[0]);

    for (size_t i = 1; i < net.pins.size(); ++i) {
        const auto& pin = net.pins[i];
        const uint64_t key = CoordKey(pin->loc.x, pin->loc.y);
        if (!seen.insert(key).second) continue;
        pins.push_back(pin);
    }
    net.pins.swap(pins);
    for (int i = 0; i < static_cast<int>(net.pins.size()); ++i) {
        net.pins[static_cast<size_t>(i)]->id = i;
    }
}

Features ComputeFeatures(const salt::Net& net, int input_degree) {
    Features out;
    out.input_degree = input_degree;
    out.degree = static_cast<int>(net.pins.size());
    if (net.pins.empty()) return out;
    int min_x = net.pins[0]->loc.x;
    int max_x = net.pins[0]->loc.x;
    int min_y = net.pins[0]->loc.y;
    int max_y = net.pins[0]->loc.y;
    for (const auto& pin : net.pins) {
        min_x = std::min(min_x, pin->loc.x);
        max_x = std::max(max_x, pin->loc.x);
        min_y = std::min(min_y, pin->loc.y);
        max_y = std::max(max_y, pin->loc.y);
    }
    out.bbox_w = max_x - min_x;
    out.bbox_h = max_y - min_y;
    return out;
}

HeaderParams ParseHeaderParams(const std::string& path) {
    std::ifstream in(path);
    if (!in.good()) throw std::runtime_error("cannot open nets file for header: " + path);
    HeaderParams params;
    std::string line;
    while (std::getline(in, line)) {
        if (line.find("unit_resistance") == 0) {
            std::istringstream iss(line);
            std::string name, colon;
            iss >> name >> colon >> params.unit_resistance;
        } else if (line.find("unit_capacitance") == 0) {
            std::istringstream iss(line);
            std::string name, colon;
            iss >> name >> colon >> params.unit_capacitance;
        } else if (line.find("driver_resistance") == 0) {
            std::istringstream iss(line);
            std::string name, colon;
            iss >> name >> colon >> params.driver_resistance;
        } else if (line == "NETS") {
            break;
        }
    }
    return params;
}

Metrics ComputeMetricsStrict(const salt::Net& net, salt::Tree& tree, const HeaderParams& params,
                             long long flute_wl, double eps) {
    Metrics out;
    const int pin_count = static_cast<int>(net.pins.size());
    if (!tree.source || pin_count <= 1) return out;

    tree.net = &net;
    const int num_nodes = tree.UpdateId();
    const auto nodes = tree.ObtainNodes();
    if (nodes.empty() || num_nodes <= 0) return out;

    std::vector<int> pin_count_seen(static_cast<size_t>(pin_count), 0);
    std::vector<long long> dist(static_cast<size_t>(pin_count), -1);
    int parent_edges = 0;
    bool parent_child_ok = true;
    bool pins_fixed = true;

    for (const auto& node : nodes) {
        if (!node) {
            parent_child_ok = false;
            continue;
        }
        if (node == tree.source) {
            if (node->parent) parent_child_ok = false;
        } else {
            if (!node->parent) parent_child_ok = false;
        }
        if (node->parent) {
            ++parent_edges;
            out.wl += Dist(node->loc, node->parent->loc);
        }
        for (const auto& child : node->children) {
            if (!child || child->parent != node) parent_child_ok = false;
        }
        if (node->pin) {
            const int id = node->pin->id;
            if (id < 0 || id >= pin_count) {
                pins_fixed = false;
                continue;
            }
            if (node->loc != net.pins[static_cast<size_t>(id)]->loc) pins_fixed = false;
            pin_count_seen[static_cast<size_t>(id)] += 1;
            DTYPE pl = 0;
            const salt::TreeNode* cur = node.get();
            while (cur && cur->parent) {
                pl += Dist(cur->loc, cur->parent->loc);
                cur = cur->parent.get();
            }
            dist[static_cast<size_t>(id)] = pl;
        }
    }

    bool covered_once = true;
    for (int id = 0; id < pin_count; ++id) {
        if (pin_count_seen[static_cast<size_t>(id)] != 1) covered_once = false;
    }

    out.valid_tree = parent_child_ok && parent_edges == static_cast<int>(nodes.size()) - 1;
    out.pins_fixed = pins_fixed;
    out.all_pins_covered_once = covered_once;

    const auto& root = net.pins[0]->loc;
    double stretch_sum = 0.0;
    int stretch_count = 0;
    out.alpha = 1.0;
    for (int pin = 1; pin < pin_count; ++pin) {
        const long long pl = dist[static_cast<size_t>(pin)];
        const auto& loc = net.pins[static_cast<size_t>(pin)]->loc;
        const long long md = std::llabs(static_cast<long long>(root.x) - loc.x) +
                             std::llabs(static_cast<long long>(root.y) - loc.y);
        out.sum_pl += pl;
        out.max_pl = std::max(out.max_pl, pl);
        out.sum_md += md;
        const double stretch = md <= 0 ? std::numeric_limits<double>::infinity()
                                       : static_cast<double>(pl) / static_cast<double>(md);
        out.alpha = std::max(out.alpha, stretch);
        stretch_sum += stretch;
        ++stretch_count;
    }
    out.nor_wl = flute_wl > 0 ? static_cast<double>(out.wl) / static_cast<double>(flute_wl)
                              : std::numeric_limits<double>::quiet_NaN();
    out.nor_path_length = out.sum_md > 0 ? static_cast<double>(out.sum_pl) / out.sum_md
                                         : std::numeric_limits<double>::quiet_NaN();
    out.avg_stretch = stretch_count > 0 ? stretch_sum / stretch_count
                                        : std::numeric_limits<double>::quiet_NaN();
    out.valid_eps = out.valid_tree && out.pins_fixed && out.all_pins_covered_once &&
                    out.alpha <= 1.0 + eps + 1e-8;
    out.valid_1p5 = out.valid_tree && out.pins_fixed && out.all_pins_covered_once &&
                    out.alpha <= 1.5 + 1e-8;

    if (out.valid_tree && num_nodes > 0 && params.unit_resistance > 0.0 &&
        params.unit_capacitance > 0.0 && params.driver_resistance > 0.0 && flute_wl > 0) {
        std::vector<double> cap(static_cast<size_t>(num_nodes), 0.0);
        tree.PostOrder([&](const std::shared_ptr<salt::TreeNode>& node) {
            if (!node || node->id < 0 || node->id >= num_nodes) return;
            if (node->pin && node != tree.source) cap[static_cast<size_t>(node->id)] = node->pin->cap;
            for (const auto& child : node->children) {
                if (!child || child->id < 0 || child->id >= num_nodes) continue;
                cap[static_cast<size_t>(node->id)] += cap[static_cast<size_t>(child->id)];
                cap[static_cast<size_t>(node->id)] +=
                    child->WireToParent() * params.unit_capacitance;
            }
        });

        std::vector<double> delay(static_cast<size_t>(num_nodes), 0.0);
        tree.PreOrder([&](const std::shared_ptr<salt::TreeNode>& node) {
            if (!node || node->id < 0 || node->id >= num_nodes) return;
            const size_t id = static_cast<size_t>(node->id);
            if (node == tree.source) {
                delay[id] = params.driver_resistance * cap[id];
            } else if (node->parent && node->parent->id >= 0 && node->parent->id < num_nodes) {
                const double d = static_cast<double>(node->WireToParent());
                delay[id] = d * params.unit_resistance *
                                (0.5 * d * params.unit_capacitance + cap[id]) +
                            delay[static_cast<size_t>(node->parent->id)];
            }
        });

        double total_cap = 0.0;
        for (const auto& pin : net.pins) total_cap += pin->cap;
        const double lb_sd =
            params.driver_resistance * (static_cast<double>(flute_wl) * params.unit_capacitance +
                                        total_cap);
        double max_lb = 0.0;
        for (int pin = 1; pin < pin_count; ++pin) {
            const auto& loc = net.pins[static_cast<size_t>(pin)]->loc;
            const double md = static_cast<double>(
                std::llabs(static_cast<long long>(root.x) - loc.x) +
                std::llabs(static_cast<long long>(root.y) - loc.y));
            const double lb = md * params.unit_resistance *
                                  (0.5 * md * params.unit_capacitance +
                                   net.pins[static_cast<size_t>(pin)]->cap) +
                              lb_sd;
            max_lb = std::max(max_lb, lb);
        }

        out.max_delay = 0.0;
        out.avg_delay = 0.0;
        int sink_count = 0;
        for (const auto& node : nodes) {
            if (!node || !node->pin || node == tree.source || node->id < 0 ||
                node->id >= num_nodes) {
                continue;
            }
            const double dly = delay[static_cast<size_t>(node->id)];
            out.max_delay = std::max(out.max_delay, dly);
            out.avg_delay += dly;
            ++sink_count;
        }
        if (sink_count > 0) out.avg_delay /= sink_count;
        if (max_lb > 0.0) {
            out.max_nor_delay = out.max_delay / max_lb;
            out.avg_nor_delay = out.avg_delay / max_lb;
        }
    }
    return out;
}

Candidate RunCandidate(const salt::Net& net, Method method, double eps, const HeaderParams& params,
                       long long flute_wl) {
    static NullBuffer null_buffer;
    static std::ostream null_stream(&null_buffer);
    Candidate out;
    salt::Tree tree;
    const auto t0 = std::chrono::high_resolution_clock::now();
    {
        ScopedCoutRedirect redirect(null_stream);
        GetATreeEx(net, tree, method, method == +Method::FLUTE ? 0.0 : eps, true, &out.stats);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    out.runtime_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    tree.RemovePhyRedundantSteiner();
    tree.RemoveTopoRedundantSteiner();
    tree.RemoveEmptyChildren();
    out.metrics = ComputeMetricsStrict(net, tree, params, flute_wl, eps);
    return out;
}

bool ParseTask(const std::string& line, Task& task) {
    std::stringstream ss(line);
    std::string field;
    if (!std::getline(ss, field, '\t')) return false;
    task.task_id = std::stoll(field);
    if (!std::getline(ss, field, '\t')) return false;
    task.eps_idx = std::stoi(field);
    if (!std::getline(ss, field, '\t')) return false;
    task.eps = std::stod(field);
    if (!std::getline(ss, task.design, '\t')) return false;
    if (!std::getline(ss, task.path, '\t')) return false;
    if (!std::getline(ss, field, '\t')) return false;
    task.offset = static_cast<std::streamoff>(std::stoll(field));
    return true;
}

void WriteHeader(std::ostream& os) {
    os << "task_id,eps_idx,eps,design,net_id,net_name,input_degree,degree,bbox_w,bbox_h,"
       << "flute_wl,flute_alpha,flute_avg_stretch,flute_valid_eps,flute_valid_1p5,"
       << "flute_tree_valid,flute_pins_fixed,flute_pins_covered,"
       << "salt_wl,far_wl,wl_reduction_pct,wl_delta,"
       << "salt_norWL,far_norWL,salt_alpha,far_alpha,salt_avg_stretch,far_avg_stretch,"
       << "salt_norPathLength,far_norPathLength,salt_maxNorDelay,far_maxNorDelay,"
       << "salt_avgNorDelay,far_avgNorDelay,salt_sum_pl,far_sum_pl,salt_max_pl,far_max_pl,"
       << "salt_sum_md,far_sum_md,salt_valid_eps,far_valid_eps,salt_valid_1p5,far_valid_1p5,"
       << "salt_tree_valid,far_tree_valid,salt_pins_fixed,far_pins_fixed,"
       << "salt_pins_covered,far_pins_covered,flute_runtime_us,salt_runtime_us,far_runtime_us,"
       << "task_wall_us,far_base,far_anchor_count,far_root_covered_count,far_critical_count,"
       << "far_repair_cost,far_base_wl,far_base_stretch,far_candidate_wl,"
       << "far_candidate_stretch,far_local_regions_tried,far_local_regions_accepted,status\n";
}

void WriteCandidateLine(std::ostream& os, const Task& task, const salt::Net& net,
                        const Features& features, long long task_wall_us,
                        const Candidate& flute, const Candidate& salt_r3,
                        const Candidate& far, const std::string& status) {
    const double red = salt_r3.metrics.wl > 0
                           ? 100.0 * static_cast<double>(salt_r3.metrics.wl - far.metrics.wl) /
                                 static_cast<double>(salt_r3.metrics.wl)
                           : std::numeric_limits<double>::quiet_NaN();
    const long long delta = salt_r3.metrics.wl - far.metrics.wl;
    os << task.task_id << "," << task.eps_idx << "," << std::setprecision(16) << task.eps
       << "," << task.design << "," << net.id << "," << net.name << ","
       << features.input_degree << "," << features.degree << "," << features.bbox_w << ","
       << features.bbox_h << "," << flute.metrics.wl << "," << flute.metrics.alpha << ","
       << flute.metrics.avg_stretch << "," << (flute.metrics.valid_eps ? 1 : 0) << ","
       << (flute.metrics.valid_1p5 ? 1 : 0) << ","
       << (flute.metrics.valid_tree ? 1 : 0) << ","
       << (flute.metrics.pins_fixed ? 1 : 0) << ","
       << (flute.metrics.all_pins_covered_once ? 1 : 0) << ","
       << salt_r3.metrics.wl << ","
       << far.metrics.wl << "," << red << "," << delta << "," << salt_r3.metrics.nor_wl
       << "," << far.metrics.nor_wl << "," << salt_r3.metrics.alpha << ","
       << far.metrics.alpha << "," << salt_r3.metrics.avg_stretch << ","
       << far.metrics.avg_stretch << "," << salt_r3.metrics.nor_path_length << ","
       << far.metrics.nor_path_length << "," << salt_r3.metrics.max_nor_delay << ","
       << far.metrics.max_nor_delay << "," << salt_r3.metrics.avg_nor_delay << ","
       << far.metrics.avg_nor_delay << "," << salt_r3.metrics.sum_pl << ","
       << far.metrics.sum_pl << "," << salt_r3.metrics.max_pl << "," << far.metrics.max_pl
       << "," << salt_r3.metrics.sum_md << "," << far.metrics.sum_md << ","
       << (salt_r3.metrics.valid_eps ? 1 : 0) << "," << (far.metrics.valid_eps ? 1 : 0)
       << "," << (salt_r3.metrics.valid_1p5 ? 1 : 0) << ","
       << (far.metrics.valid_1p5 ? 1 : 0) << ","
       << (salt_r3.metrics.valid_tree ? 1 : 0) << ","
       << (far.metrics.valid_tree ? 1 : 0) << ","
       << (salt_r3.metrics.pins_fixed ? 1 : 0) << ","
       << (far.metrics.pins_fixed ? 1 : 0) << ","
       << (salt_r3.metrics.all_pins_covered_once ? 1 : 0) << ","
       << (far.metrics.all_pins_covered_once ? 1 : 0) << "," << flute.runtime_us << ","
       << salt_r3.runtime_us << "," << far.runtime_us << "," << task_wall_us << ",";

    if (far.stats.far_has) {
        os << far.stats.far_base << "," << far.stats.far_anchor_count << ","
           << far.stats.far_root_covered_count << "," << far.stats.far_critical_count
           << "," << far.stats.far_repair_cost << "," << far.stats.far_base_wl << ","
           << far.stats.far_base_stretch << "," << far.stats.far_candidate_wl << ","
           << far.stats.far_candidate_stretch << ","
           << far.stats.far_local_regions_tried << ","
           << far.stats.far_local_regions_accepted;
    } else {
        os << ",,,,,,,,,,";
    }
    os << "," << status << "\n";
}

std::ifstream& OpenCached(std::unordered_map<std::string, std::unique_ptr<std::ifstream>>& cache,
                          const std::string& path) {
    auto it = cache.find(path);
    if (it == cache.end()) {
        std::unique_ptr<std::ifstream> stream(new std::ifstream(path));
        if (!stream->good()) {
            throw std::runtime_error("cannot open nets file: " + path);
        }
        it = cache.emplace(path, std::move(stream)).first;
    }
    return *it->second;
}

const HeaderParams& HeaderCached(std::unordered_map<std::string, HeaderParams>& cache,
                                 const std::string& path) {
    auto it = cache.find(path);
    if (it == cache.end()) {
        it = cache.emplace(path, ParseHeaderParams(path)).first;
    }
    return it->second;
}

}  // namespace

int main(int argc, char** argv) {
    PrintThirdPartyAttribution();

    bool write_header = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--header") {
            write_header = true;
        }
    }
    if (write_header) {
        WriteHeader(std::cout);
        return 0;
    }

    std::unordered_map<std::string, std::unique_ptr<std::ifstream>> files;
    std::unordered_map<std::string, HeaderParams> headers;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        if (line == "__quit__") break;
        Task task;
        try {
            if (!ParseTask(line, task)) {
                std::cout << "-1,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,parse_error\n";
                std::cout.flush();
                continue;
            }
            const auto t0 = std::chrono::high_resolution_clock::now();
            const HeaderParams& params = HeaderCached(headers, task.path);
            std::ifstream& is = OpenCached(files, task.path);
            is.clear();
            is.seekg(task.offset);
            salt::Net net;
            if (!net.Read(is)) {
                std::cout << task.task_id << "," << task.eps_idx << "," << task.eps << ","
                          << task.design
                          << ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,read_error\n";
                std::cout.flush();
                continue;
            }
            const int input_degree = static_cast<int>(net.pins.size());
            DedupPreserveInputSource(net);
            const Features features = ComputeFeatures(net, input_degree);

            Candidate flute = RunCandidate(net, Method::FLUTE, task.eps, params, 0);
            const long long flute_wl = flute.metrics.wl;
            flute.metrics.nor_wl = flute_wl > 0 ? 1.0 : std::numeric_limits<double>::quiet_NaN();

            Candidate salt_r3 = RunCandidate(net, Method::SALT_R3, task.eps, params, flute_wl);
            Candidate far = RunCandidate(net, Method::FAR_SALT_FULL, task.eps, params, flute_wl);
            const auto t1 = std::chrono::high_resolution_clock::now();
            const long long task_wall_us =
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            WriteCandidateLine(std::cout, task, net, features, task_wall_us, flute, salt_r3, far,
                               "ok");
            std::cout.flush();
        } catch (const std::exception& ex) {
            std::string design = task.design.empty() ? "" : task.design;
            std::cout << task.task_id << "," << task.eps_idx << "," << task.eps << ","
                      << design
                      << ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,exception\n";
            std::cout.flush();
            std::cerr << "task " << task.task_id << " failed: " << ex.what() << "\n";
        }
    }
    return 0;
}
