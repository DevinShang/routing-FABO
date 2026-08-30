#include "ust_salt.h"

#include "base/eval.h"
#include "base/flute.h"
#include "base/flute/flute.h"
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif
#ifdef abs
#undef abs
#endif
#include "base/mst.h"
#include "base/rsa.h"
#include "refine/refine.h"
#include "salt.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <limits>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace salt {
namespace {

constexpr double kInf = 1e100;
constexpr double kTol = 1e-9;

struct FarProfileRun {
    int net_id = -1;
    int degree = 0;
    bool full_mode = false;
    std::chrono::high_resolution_clock::time_point start;
    std::unordered_map<std::string, long long> phase_us;
    std::unordered_map<std::string, int> phase_count;
};

FarProfileRun*& FarProfileCurrent() {
    static FarProfileRun* current = nullptr;
    return current;
}

std::vector<std::string>& FarProfileContextStack() {
    static std::vector<std::string> stack;
    return stack;
}

bool FarProfileEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("FAR_PROFILE_PHASES");
        return value && std::atoi(value) != 0;
    }();
    return enabled;
}

std::ostream& FarProfileStream() {
    static std::ofstream ofs;
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        const char* path = std::getenv("FAR_PROFILE_OUT");
        if (path && *path) ofs.open(path, std::ios::out | std::ios::app);
    }
    return ofs.is_open() ? ofs : std::cerr;
}

std::string FarProfileLabel(const char* label) {
    const auto& stack = FarProfileContextStack();
    if (stack.empty()) return std::string(label);
    std::string out;
    for (const auto& prefix : stack) {
        if (!out.empty()) out += '.';
        out += prefix;
    }
    out += '.';
    out += label;
    return out;
}

void FarProfileAdd(const std::string& label, long long us) {
    FarProfileRun* current = FarProfileCurrent();
    if (!current) return;
    current->phase_us[label] += us;
    current->phase_count[label] += 1;
}

struct FarProfileScope {
    explicit FarProfileScope(const char* label)
        : active(FarProfileEnabled() && FarProfileCurrent()),
          phase(active ? FarProfileLabel(label) : std::string()),
          start(active ? std::chrono::high_resolution_clock::now() :
                         std::chrono::high_resolution_clock::time_point{}) {}

    ~FarProfileScope() {
        if (!active) return;
        const auto end = std::chrono::high_resolution_clock::now();
        const long long us =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        FarProfileAdd(phase, us);
    }

    bool active = false;
    std::string phase;
    std::chrono::high_resolution_clock::time_point start;
};

struct FarProfileContextScope {
    explicit FarProfileContextScope(const char* prefix)
        : active(FarProfileEnabled() && FarProfileCurrent()) {
        if (active) FarProfileContextStack().push_back(prefix);
    }

    ~FarProfileContextScope() {
        if (active) FarProfileContextStack().pop_back();
    }

    bool active = false;
};

void FarProfileEmit(const FarProfileRun& run) {
    std::vector<std::string> keys;
    keys.reserve(run.phase_us.size());
    for (const auto& kv : run.phase_us) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());

    std::ostream& os = FarProfileStream();
    os << "{\"event\":\"far_phase_profile\""
       << ",\"net_id\":" << run.net_id
       << ",\"degree\":" << run.degree
       << ",\"full_mode\":" << (run.full_mode ? 1 : 0)
       << ",\"phase_us\":{";
    for (size_t i = 0; i < keys.size(); ++i) {
        if (i) os << ',';
        os << "\"" << keys[i] << "\":" << run.phase_us.at(keys[i]);
    }
    os << "},\"phase_count\":{";
    for (size_t i = 0; i < keys.size(); ++i) {
        if (i) os << ',';
        os << "\"" << keys[i] << "\":" << run.phase_count.at(keys[i]);
    }
    os << "}}\n";
}

struct FarProfileRunScope {
    FarProfileRun run;
    FarProfileRun* previous = nullptr;
    bool active = false;

    FarProfileRunScope(const Net& net, bool full_mode) : active(FarProfileEnabled()) {
        if (!active) return;
        run.net_id = net.id;
        run.degree = static_cast<int>(net.pins.size());
        run.full_mode = full_mode;
        run.start = std::chrono::high_resolution_clock::now();
        previous = FarProfileCurrent();
        FarProfileCurrent() = &run;
    }

    ~FarProfileRunScope() {
        if (!active) return;
        const auto end = std::chrono::high_resolution_clock::now();
        run.phase_us["run.total"] +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - run.start).count();
        run.phase_count["run.total"] += 1;
        FarProfileEmit(run);
        FarProfileCurrent() = previous;
    }
};

enum class ConnectorVariant : unsigned char { Rsa = 0, EsMst = 1 };

struct SupportTree {
    std::string name;
    double param = 0.0;
    int root = 0;
    int n = 0;
    DTYPE total_w = 0;
    std::vector<Point> loc;
    std::vector<std::shared_ptr<Pin>> pin_ref;
    std::vector<int> pin_to_node;
    std::vector<int> parent;
    std::vector<DTYPE> edge_w;
    std::vector<DTYPE> root_dist;
    std::vector<int> tin;
    std::vector<int> tout;
    std::vector<int> preorder;
    std::vector<int> postorder;
    std::vector<std::vector<int>> children;
    std::vector<std::vector<std::pair<int, DTYPE>>> adj;

    bool InSubtree(int anc, int v) const { return tin[anc] <= tin[v] && tin[v] <= tout[anc]; }
};

struct DpSolution {
    std::vector<int> block_center;
    std::vector<unsigned char> is_center;
};

struct CenterTerminal {
    int support_node = -1;
    Point loc;
    std::shared_ptr<Pin> original_pin;
};

struct CenterNetData {
    Net net;
    std::vector<CenterTerminal> terminals;
    std::vector<int> support_to_clone;
};

struct CandidateResult {
    Tree tree;
    bool valid = false;
    DTYPE wl = std::numeric_limits<DTYPE>::max();
    double max_stretch = std::numeric_limits<double>::infinity();
};

void BuildBestRealizedCandidate(
    const Net& net,
    const SupportTree& support,
    const DpSolution& solution,
    double eps,
    CandidateResult& best
);

int SerrEnvInt(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed < 0 || parsed > 1000000) return fallback;
    return static_cast<int>(parsed);
}

bool FarHighDegreeFastMode(const Net& net) {
    if (SerrEnvInt("FAR_HD_FAST_MODE", 0) == 0) return false;
    const int threshold = std::max(1, SerrEnvInt("FAR_HD_FAST_THRESHOLD", 256));
    return static_cast<int>(net.pins.size()) >= threshold;
}

DTYPE EdgeWeight(const Point& a, const Point& b) { return Dist(a, b); }

void MoveTree(Tree& dst, Tree& src) {
    dst.Reset();
    dst.source = src.source;
    dst.net = src.net;
    src.source = nullptr;
    src.net = nullptr;
}

std::shared_ptr<TreeNode> CloneTreeNodeForNet(const std::shared_ptr<TreeNode>& node) {
    if (!node) return nullptr;
    auto clone = node->pin ? std::make_shared<TreeNode>(node->pin) : std::make_shared<TreeNode>(node->loc);
    clone->loc = node->loc;
    clone->id = node->id;
    clone->children.reserve(node->children.size());
    for (const auto& child : node->children) {
        auto child_clone = CloneTreeNodeForNet(child);
        TreeNode::SetParent(child_clone, clone);
    }
    return clone;
}

void CopyTreeForNet(Tree& dst, const Tree& src, const Net& net) {
    dst.Reset();
    dst.net = &net;
    dst.source = CloneTreeNodeForNet(src.source);
}

bool SupportsEquivalent(const SupportTree& lhs, const SupportTree& rhs) {
    if (lhs.n != rhs.n) return false;
    if (lhs.parent != rhs.parent) return false;
    return lhs.loc == rhs.loc;
}

SupportTree BuildSupportTreeFromArrays(
    const std::vector<Point>& loc,
    const std::vector<std::shared_ptr<Pin>>& pin_ref,
    const std::vector<int>& pin_to_node,
    const std::vector<int>& parent,
    const std::string& name,
    double param
) {
    FarProfileScope profile_total("support.arrays_total");
    SupportTree out;
    out.name = name;
    out.param = param;
    out.n = static_cast<int>(parent.size());
    out.loc = loc;
    out.pin_ref = pin_ref;
    out.pin_to_node = pin_to_node;
    out.parent = parent;
    out.edge_w.assign(static_cast<size_t>(out.n), 0);
    out.root_dist.assign(static_cast<size_t>(out.n), 0);
    out.tin.assign(static_cast<size_t>(out.n), 0);
    out.tout.assign(static_cast<size_t>(out.n), 0);
    out.children.assign(static_cast<size_t>(out.n), {});
    out.adj.assign(static_cast<size_t>(out.n), {});

    {
        FarProfileScope profile("support.arrays_build_adj");
        for (int v = 0; v < out.n; ++v) {
            const int p = parent[static_cast<size_t>(v)];
            if (p < 0) {
                out.root = v;
                continue;
            }
            const DTYPE w = EdgeWeight(out.loc[static_cast<size_t>(v)], out.loc[static_cast<size_t>(p)]);
            out.edge_w[static_cast<size_t>(v)] = w;
            out.total_w += w;
            out.children[static_cast<size_t>(p)].push_back(v);
            out.adj[static_cast<size_t>(v)].push_back({p, w});
            out.adj[static_cast<size_t>(p)].push_back({v, w});
        }

        for (auto& ch : out.children) {
            std::sort(ch.begin(), ch.end());
        }
    }

    {
        FarProfileScope profile("support.arrays_dfs");
        int timer = 0;
        auto dfs = [&](auto&& self, int v) -> void {
            out.tin[static_cast<size_t>(v)] = timer++;
            out.preorder.push_back(v);
            for (int u : out.children[static_cast<size_t>(v)]) {
                out.root_dist[static_cast<size_t>(u)] = out.root_dist[static_cast<size_t>(v)] + out.edge_w[static_cast<size_t>(u)];
                self(self, u);
            }
            out.tout[static_cast<size_t>(v)] = timer - 1;
            out.postorder.push_back(v);
        };
        dfs(dfs, out.root);
    }

    return out;
}

SupportTree BuildSupportFromTree(Tree& tree, const std::string& name, double param) {
    FarProfileScope profile("support.from_tree");
    tree.RemovePhyRedundantSteiner();
    tree.RemoveTopoRedundantSteiner();
    tree.RemoveEmptyChildren();
    tree.UpdateId();
    const auto nodes = tree.ObtainNodes();
    const int n = static_cast<int>(nodes.size());

    std::vector<Point> loc(static_cast<size_t>(n));
    std::vector<std::shared_ptr<Pin>> pin_ref(static_cast<size_t>(n), nullptr);
    std::vector<int> pin_to_node(tree.net->pins.size(), -1);
    std::vector<int> parent(static_cast<size_t>(n), -1);

    for (const auto& node : nodes) {
        const int v = node->id;
        loc[static_cast<size_t>(v)] = node->loc;
        pin_ref[static_cast<size_t>(v)] = node->pin;
        if (node->pin) {
            pin_to_node[static_cast<size_t>(node->pin->id)] = v;
        }
        if (node->parent) {
            parent[static_cast<size_t>(v)] = node->parent->id;
        }
    }

    return BuildSupportTreeFromArrays(loc, pin_ref, pin_to_node, parent, name, param);
}

SupportTree BuildFluteSupport(const Net& net, bool refine_support) {
    FarProfileContextScope profile_ctx(refine_support ? "support.flute_r1" : "support.flute_r0");
    Tree smt;
    {
        FarProfileScope profile("flute_run");
        FluteBuilder flute_builder;
        flute_builder.Run(net, smt);
    }
    if (refine_support) {
        FarProfileScope profile("refine_support");
        Refine::Flip(smt);
        Refine::UShift(smt);
    }
    return BuildSupportFromTree(smt, refine_support ? "FLUTE_R1" : "FLUTE_R0", refine_support ? 1.0 : 0.0);
}

struct LarsCoordHash {
    std::size_t operator()(const std::pair<DTYPE, DTYPE>& key) const {
        const auto ux = static_cast<std::uint32_t>(key.first);
        const auto uy = static_cast<std::uint32_t>(key.second);
        return static_cast<std::size_t>((static_cast<std::uint64_t>(ux) << 32) ^ static_cast<std::uint64_t>(uy));
    }
};

SupportTree BuildFluteSupportWithAccuracy(const Net& net, int accuracy, bool refine_support) {
    FarProfileContextScope profile_ctx(refine_support ? "support.flute_a_refined" : "support.flute_a");
    static bool once = false;
    if (!once) {
        FarProfileScope profile("read_lut");
        flute::readLUT();
        once = true;
    }

    const int d = static_cast<int>(net.pins.size());
    assert(d <= MAXD);
    DTYPE x[MAXD];
    DTYPE y[MAXD];
    for (int i = 0; i < d; ++i) {
        x[i] = net.pins[static_cast<size_t>(i)]->loc.x;
        y[i] = net.pins[static_cast<size_t>(i)]->loc.y;
    }
    flute::Tree flute_tree;
    {
        FarProfileScope profile("flute_run");
        flute_tree = flute::flute(d, x, y, accuracy);
    }

    std::unordered_map<std::pair<DTYPE, DTYPE>, std::shared_ptr<TreeNode>, LarsCoordHash> key2node;
    for (const auto& pin : net.pins) key2node[{pin->loc.x, pin->loc.y}] = std::make_shared<TreeNode>(pin);

    auto find_or_create = [&](DTYPE px, DTYPE py) {
        auto it = key2node.find({px, py});
        if (it != key2node.end()) return it->second;
        auto node = std::make_shared<TreeNode>(px, py);
        key2node[{px, py}] = node;
        return node;
    };

    for (int i = 0; i < 2 * flute_tree.deg - 2; ++i) {
        const int j = flute_tree.branch[i].n;
        if (flute_tree.branch[i].x == flute_tree.branch[j].x && flute_tree.branch[i].y == flute_tree.branch[j].y) continue;
        auto a = find_or_create(flute_tree.branch[i].x, flute_tree.branch[i].y);
        auto b = find_or_create(flute_tree.branch[j].x, flute_tree.branch[j].y);
        a->children.push_back(b);
        b->children.push_back(a);
    }

    Tree smt;
    smt.source = key2node[{net.source()->loc.x, net.source()->loc.y}];
    smt.SetParentFromUndirectedAdjList();
    smt.net = &net;
    std::free(flute_tree.branch);

    if (refine_support) {
        FarProfileScope profile("refine_support");
        Refine::Flip(smt);
        Refine::UShift(smt);
    }
    std::ostringstream name;
    name << "FLUTE_A" << accuracy << (refine_support ? "_R1" : "_R0");
    return BuildSupportFromTree(smt, name.str(), static_cast<double>(accuracy) + (refine_support ? 0.1 : 0.0));
}

DTYPE MaxOverlap(DTYPE z1, DTYPE z2) {
    if (z1 >= 0 && z2 >= 0) return std::min(z1, z2);
    if (z1 <= 0 && z2 <= 0) return std::max(z1, z2);
    return 0;
}

void BuildEsConnectorMstOrder(Net& net, Tree& tree) {
    if (net.pins.size() == 1) {
        tree.source = std::make_shared<TreeNode>(net.source());
        tree.net = &net;
        return;
    }

    const auto ori_src = net.source()->loc;
    for (auto& p : net.pins) p->loc -= ori_src;

    // Use the ES-RSA variant order: MST preorder (excluding the source).
    Tree mst;
    MstBuilder mst_builder;
    mst_builder.Run(net, mst);
    std::vector<std::shared_ptr<TreeNode>> nodes;
    mst.PreOrder([&](const std::shared_ptr<TreeNode>& tn) {
        if (!tn->parent) return;
        nodes.push_back(std::make_shared<TreeNode>(tn->pin->loc, tn->pin, tn->pin->id));
    });

    while (nodes.size() > 1) {
        std::vector<std::shared_ptr<TreeNode>> next;
        for (size_t i = 1; i < nodes.size(); i += 2) {
            auto a = nodes[i - 1];
            auto b = nodes[i];
            Point p;
            p.x = MaxOverlap(a->loc.x, b->loc.x);
            p.y = MaxOverlap(a->loc.y, b->loc.y);
            if (p == a->loc) {
                TreeNode::SetParent(b, a);
                next.push_back(a);
            } else if (p == b->loc) {
                TreeNode::SetParent(a, b);
                next.push_back(b);
            } else {
                auto steiner = std::make_shared<TreeNode>(p);
                TreeNode::SetParent(a, steiner);
                TreeNode::SetParent(b, steiner);
                next.push_back(steiner);
            }
        }
        if (nodes.size() % 2 == 1) next.push_back(nodes.back());
        nodes.swap(next);
    }

    auto last = nodes.front();
    if (last->loc == net.source()->loc) {
        tree.source = last;
        tree.source->pin = net.source();
        tree.source->id = 0;
    } else {
        tree.source = std::make_shared<TreeNode>(net.source());
        TreeNode::SetParent(last, tree.source);
    }

    for (auto& p : net.pins) p->loc += ori_src;
    tree.PreOrder([&](const std::shared_ptr<TreeNode>& node) { node->loc += ori_src; });
    tree.net = &net;
}

CenterNetData MakeCenterNet(const SupportTree& support, const std::vector<int>& requested_order) {
    CenterNetData out;
    out.net.id = 0;
    out.net.name = support.name;
    out.net.withCap = false;
    out.support_to_clone.assign(static_cast<size_t>(support.n), -1);

    std::vector<int> order;
    order.reserve(requested_order.size() + 1);
    std::vector<unsigned char> seen(static_cast<size_t>(support.n), 0);
    order.push_back(support.root);
    seen[static_cast<size_t>(support.root)] = 1;
    for (int v : requested_order) {
        if (v < 0 || v >= support.n) continue;
        if (seen[static_cast<size_t>(v)]) continue;
        seen[static_cast<size_t>(v)] = 1;
        order.push_back(v);
    }

    out.terminals.reserve(order.size());
    out.net.pins.reserve(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        const int support_node = order[i];
        out.support_to_clone[static_cast<size_t>(support_node)] = static_cast<int>(i);
        CenterTerminal terminal;
        terminal.support_node = support_node;
        terminal.loc = support.loc[static_cast<size_t>(support_node)];
        terminal.original_pin = support.pin_ref[static_cast<size_t>(support_node)];
        out.terminals.push_back(terminal);
        out.net.pins.push_back(std::make_shared<Pin>(terminal.loc, static_cast<int>(i), terminal.original_pin ? terminal.original_pin->cap : 0.0));
    }
    return out;
}

void RemapConnectorTerminals(Tree& tree, const CenterNetData& center_net) {
    tree.PreOrder([&](const std::shared_ptr<TreeNode>& node) {
        if (!node->pin) return;
        const int clone_id = node->pin->id;
        const auto& terminal = center_net.terminals[static_cast<size_t>(clone_id)];
        if (terminal.original_pin) {
            node->pin = terminal.original_pin;
            node->id = terminal.original_pin->id;
        } else {
            node->pin = nullptr;
            node->id = -1;
            node->loc = terminal.loc;
        }
    });
}

std::vector<std::shared_ptr<TreeNode>> CollectPinNodes(Tree& tree, int num_pins) {
    std::vector<std::shared_ptr<TreeNode>> nodes(static_cast<size_t>(num_pins), nullptr);
    tree.PreOrder([&](const std::shared_ptr<TreeNode>& node) {
        if (!node->pin) return;
        const int id = node->pin->id;
        if (id >= 0 && id < num_pins) nodes[static_cast<size_t>(id)] = node;
    });
    return nodes;
}

std::vector<int> CenterListFromFlags(const std::vector<unsigned char>& is_center) {
    std::vector<int> centers;
    centers.reserve(is_center.size());
    for (size_t i = 0; i < is_center.size(); ++i) {
        if (is_center[i]) centers.push_back(static_cast<int>(i));
    }
    return centers;
}

std::vector<unsigned char> SubtreeHasCenter(const SupportTree& support, const std::vector<unsigned char>& is_center) {
    std::vector<unsigned char> has_center = is_center;
    for (int v : support.postorder) {
        for (int u : support.children[static_cast<size_t>(v)]) {
            if (has_center[static_cast<size_t>(u)]) has_center[static_cast<size_t>(v)] = 1;
        }
    }
    return has_center;
}

std::vector<int> BuildBackboneOrder(const SupportTree& support, const std::vector<unsigned char>& is_center) {
    std::vector<int> order;
    order.reserve(is_center.size());
    order.push_back(support.root);
    const auto has_center = SubtreeHasCenter(support, is_center);
    auto dfs = [&](auto&& self, int v) -> void {
        if (v != support.root && is_center[static_cast<size_t>(v)]) order.push_back(v);
        for (int u : support.children[static_cast<size_t>(v)]) {
            if (has_center[static_cast<size_t>(u)]) self(self, u);
        }
    };
    dfs(dfs, support.root);
    return order;
}

std::vector<int> BuildCenterMstOrder(const SupportTree& support, const std::vector<int>& centers) {
    if (centers.size() <= 1) return {support.root};
    auto center_net = MakeCenterNet(support, centers);
    Tree mst;
    MstBuilder mst_builder;
    mst_builder.Run(center_net.net, mst);
    std::vector<int> order;
    order.reserve(centers.size());
    order.push_back(support.root);
    mst.PreOrder([&](const std::shared_ptr<TreeNode>& node) {
        if (!node->pin || node->pin->id == 0) return;
        order.push_back(center_net.terminals[static_cast<size_t>(node->pin->id)].support_node);
    });
    return order;
}

void AppendUniqueOrder(std::vector<std::vector<int>>& orders, const std::vector<int>& order) {
    if (order.empty()) return;
    for (const auto& existing : orders) {
        if (existing == order) return;
    }
    orders.push_back(order);
}

void BuildConnector(const CenterNetData& center_net, ConnectorVariant variant, double eps, Tree& connector) {
    if (variant == ConnectorVariant::Rsa) {
        RsaBuilder builder;
        builder.Run(center_net.net, connector);
    } else {
        (void)eps;
        BuildEsConnectorMstOrder(const_cast<Net&>(center_net.net), connector);
    }
}

void BuildRealizedTreeFromCenterNet(
    const Net& net,
    const SupportTree& support,
    const std::vector<int>& block_center,
    const std::vector<unsigned char>& is_center,
    const CenterNetData& center_net,
    ConnectorVariant variant,
    double eps,
    Tree& out
) {
    FarProfileScope profile_total("realize.from_center_net");
    Tree connector;
    {
        FarProfileScope profile("realize.connector");
        BuildConnector(center_net, variant, eps, connector);
    }

    std::vector<std::shared_ptr<TreeNode>> center_nodes;
    {
        FarProfileScope profile("realize.collect_remap");
        center_nodes = CollectPinNodes(connector, static_cast<int>(center_net.net.pins.size()));
        RemapConnectorTerminals(connector, center_net);
    }

    auto make_support_node = [&](int v) {
        return std::make_shared<TreeNode>(
            support.loc[static_cast<size_t>(v)],
            support.pin_ref[static_cast<size_t>(v)],
            support.pin_ref[static_cast<size_t>(v)] ? support.pin_ref[static_cast<size_t>(v)]->id : -1
        );
    };

    auto attach_block =
        [&](auto&& self, int v, int p, int center, const std::shared_ptr<TreeNode>& cur_node) -> void {
            for (const auto& edge : support.adj[static_cast<size_t>(v)]) {
                const int u = edge.first;
                if (u == p) continue;
                if (block_center[static_cast<size_t>(u)] != center) continue;
                if (u != center && is_center[static_cast<size_t>(u)]) continue;
                auto child = make_support_node(u);
                TreeNode::SetParent(child, cur_node);
                self(self, u, v, center, child);
            }
        };

    {
        FarProfileScope profile("realize.attach_blocks");
        const auto centers = CenterListFromFlags(is_center);
        for (int center : centers) {
            const int clone_id = center_net.support_to_clone[static_cast<size_t>(center)];
            if (clone_id < 0 || clone_id >= static_cast<int>(center_nodes.size()) || !center_nodes[static_cast<size_t>(clone_id)]) {
                out.Reset();
                return;
            }
            attach_block(attach_block, center, -1, center, center_nodes[static_cast<size_t>(clone_id)]);
        }
    }

    {
        FarProfileScope profile("realize.cleanup_move");
        connector.net = &net;
        connector.RemoveEmptyChildren();
        connector.RemovePhyRedundantSteiner();
        connector.RemoveTopoRedundantSteiner();
        MoveTree(out, connector);
        out.net = &net;
    }
}

void ApplyRefineMask(Tree& tree, double eps, int mask, int substitute_max_rounds,
                     int substitute_use_rtree = -1,
                     int substitute_linear_candidate_cap = 0,
                     int substitute_linear_query_mode = 0) {
    if (mask & 1) {
        FarProfileScope profile("refine.cancel_intersect");
        Refine::CancelIntersect(tree);
    }
    if (mask & 2) {
        FarProfileScope profile("refine.flip");
        Refine::Flip(tree);
    }
    if (mask & 4) {
        FarProfileScope profile("refine.ushift");
        Refine::UShift(tree);
    }
    if (mask & 8) {
        FarProfileScope profile("refine.substitute");
        const bool use_rtree = substitute_use_rtree >= 0 ? substitute_use_rtree != 0 : true;
        Refine::Substitute(tree, eps, use_rtree, substitute_max_rounds,
                           substitute_linear_candidate_cap,
                           substitute_linear_query_mode);
    }
    if (mask & 16) {
        FarProfileScope profile("refine.cleanup");
        tree.RemovePhyRedundantSteiner();
        tree.RemoveTopoRedundantSteiner();
    }
}

void ApplyFullRefine(Tree& tree, double eps) {
    const int mask = SerrEnvInt("FAR_FULL_REFINE_MASK", 31);
    const int substitute_max_rounds = std::max(0, SerrEnvInt("FAR_SUBSTITUTE_MAX_ROUNDS", 0));
    ApplyRefineMask(tree, eps, mask, substitute_max_rounds);
}

struct FarFastRealizerOptions {
    int refine_count = -1;
    int refine_mask = -1;
    int substitute_max_rounds = -1;
    int substitute_use_rtree = -1;
    int substitute_linear_candidate_cap = 0;
    int substitute_linear_query_mode = 0;
    int variant_mode = -1;
};

bool CheckCoverageAndParents(const Tree& tree, const Net& net) {
    if (!tree.source) return false;

    std::vector<unsigned char> seen_pin(net.pins.size(), 0);
    std::vector<const TreeNode*> seen_node;
    seen_node.reserve(std::max<size_t>(16, net.pins.size() * 3));
    std::vector<const TreeNode*> stack;
    stack.reserve(std::max<size_t>(16, net.pins.size() * 3));
    stack.push_back(tree.source.get());
    while (!stack.empty()) {
        const TreeNode* node = stack.back();
        stack.pop_back();
        if (!node) return false;
        if (std::find(seen_node.begin(), seen_node.end(), node) != seen_node.end()) return false;
        seen_node.push_back(node);
        if (node->pin) {
            const int id = node->pin->id;
            if (id < 0 || id >= static_cast<int>(net.pins.size()) || seen_pin[static_cast<size_t>(id)]) {
                return false;
            }
            seen_pin[static_cast<size_t>(id)] = 1;
        }
        for (const auto& child : node->children) {
            if (!child || child->parent.get() != node) return false;
            stack.push_back(child.get());
        }
    }

    for (unsigned char flag : seen_pin) {
        if (!flag) return false;
    }
    return true;
}

void EvaluateTreeMetricsInPlace(Tree& tree, const Net& net, double eps, CandidateResult& out) {
    FarProfileScope profile("eval.metrics_in_place");
    out = CandidateResult{};
    tree.net = &net;
    if (!CheckCoverageAndParents(tree, net)) return;
    tree.RemoveEmptyChildren();
    tree.RemovePhyRedundantSteiner();
    tree.RemoveTopoRedundantSteiner();
    if (!CheckCoverageAndParents(tree, net)) return;
    tree.UpdateId();
    WireLengthEval eval(tree);
    if (eval.maxStretch > 1.0 + eps + 1e-8) return;
    out.valid = true;
    out.wl = eval.wireLength;
    out.max_stretch = eval.maxStretch;
}

void EvaluateTree(Tree& tree, const Net& net, double eps, CandidateResult& out) {
    FarProfileScope profile("eval.evaluate_tree");
    EvaluateTreeMetricsInPlace(tree, net, eps, out);
    if (!out.valid) return;
    MoveTree(out.tree, tree);
}

void ApplySuperTailSequence(Tree& tree, double eps, int rounds, int mask_override = -1) {
    FarProfileScope profile_total("super_tail.sequence_total");
    const int mask = (mask_override >= 0) ? mask_override : SerrEnvInt("FAR_SUPER_TAIL_MASK", 511);
    const bool skip_round_cleanup = SerrEnvInt("FAR_SUPER_TAIL_SKIP_ROUND_CLEANUP", 0) != 0;
    const int substitute_max_rounds = std::max(0, SerrEnvInt("FAR_SUBSTITUTE_MAX_ROUNDS", 0));
    for (int round = 0; round < rounds; ++round) {
        if (mask & 1) {
            FarProfileScope profile("super_tail.quad1");
            Refine::QuadBundleSubstitute(tree, eps);
        }
        if (mask & 2) {
            FarProfileScope profile("super_tail.triplet1");
            Refine::TripletBundleSubstitute(tree, eps);
        }
        if (mask & 4) {
            FarProfileScope profile("super_tail.sibling_pair1");
            Refine::SiblingPairSubstitute(tree, eps, true);
        }
        if (mask & 8) {
            FarProfileScope profile("super_tail.substitute1");
            Refine::Substitute(tree, eps, true, substitute_max_rounds);
        }
        if (mask & 16) {
            FarProfileScope profile("super_tail.cancel_flip_ushift");
            Refine::CancelIntersect(tree);
            Refine::Flip(tree);
            Refine::UShift(tree);
        }
        if (mask & 32) {
            FarProfileScope profile("super_tail.quad2");
            Refine::QuadBundleSubstitute(tree, eps);
        }
        if (mask & 64) {
            FarProfileScope profile("super_tail.triplet2");
            Refine::TripletBundleSubstitute(tree, eps);
        }
        if (mask & 128) {
            FarProfileScope profile("super_tail.sibling_pair2");
            Refine::SiblingPairSubstitute(tree, eps, true);
        }
        if (mask & 256) {
            FarProfileScope profile("super_tail.substitute2");
            Refine::Substitute(tree, eps, true, substitute_max_rounds);
        }
        if (!skip_round_cleanup) {
            FarProfileScope profile("super_tail.cleanup");
            tree.RemovePhyRedundantSteiner();
            tree.RemoveTopoRedundantSteiner();
            tree.RemoveEmptyChildren();
            tree.QuickCheck();
        }
    }
}

void BuildSuperTailCandidate(const Net& net,
                             const CandidateResult& incumbent,
                             double eps,
                             int rounds,
                             CandidateResult& out,
                             int mask_override = -1) {
    FarProfileScope profile_total("super_tail.candidate_total");
    out = CandidateResult{};
    if (!incumbent.valid || rounds <= 0) return;
    Tree trial;
    CopyTreeForNet(trial, incumbent.tree, net);
    ApplySuperTailSequence(trial, eps, rounds, mask_override);
    EvaluateTree(trial, net, eps, out);
}

bool BetterCandidate(const CandidateResult& lhs, const CandidateResult& rhs) {
    if (!lhs.valid) return false;
    if (!rhs.valid) return true;
    if (lhs.wl != rhs.wl) return lhs.wl < rhs.wl;
    return lhs.max_stretch < rhs.max_stretch - kTol;
}

bool ValidatePartition(const SupportTree& support, const DpSolution& solution) {
    const int n = support.n;
    if (static_cast<int>(solution.block_center.size()) != n || static_cast<int>(solution.is_center.size()) != n) {
        return false;
    }

    std::vector<int> counts(static_cast<size_t>(n), 0);
    for (int v = 0; v < n; ++v) {
        const int c = solution.block_center[static_cast<size_t>(v)];
        if (c < 0 || c >= n) return false;
        ++counts[static_cast<size_t>(c)];
    }

    for (int c = 0; c < n; ++c) {
        const bool center_flag = solution.is_center[static_cast<size_t>(c)] != 0;
        if (center_flag != (counts[static_cast<size_t>(c)] > 0)) return false;
        if (center_flag && solution.block_center[static_cast<size_t>(c)] != c) return false;
    }
    if (!solution.is_center[static_cast<size_t>(support.root)]) return false;

    std::vector<unsigned char> visited(static_cast<size_t>(n), 0);
    for (int c = 0; c < n; ++c) {
        if (!solution.is_center[static_cast<size_t>(c)]) continue;
        std::vector<int> stack{c};
        visited[static_cast<size_t>(c)] = 1;
        while (!stack.empty()) {
            const int v = stack.back();
            stack.pop_back();
            for (const auto& edge : support.adj[static_cast<size_t>(v)]) {
                const int u = edge.first;
                if (visited[static_cast<size_t>(u)]) continue;
                if (solution.block_center[static_cast<size_t>(u)] != c) continue;
                visited[static_cast<size_t>(u)] = 1;
                stack.push_back(u);
            }
        }
        for (int v = 0; v < n; ++v) {
            if (solution.block_center[static_cast<size_t>(v)] == c && !visited[static_cast<size_t>(v)]) return false;
        }
    }
    return true;
}

void BuildBestRealizedCandidate(
    const Net& net,
    const SupportTree& support,
    const DpSolution& solution,
    double eps,
    CandidateResult& best
) {
    FarProfileScope profile_total("realizer.full_total");
    best = CandidateResult{};
    const auto centers = CenterListFromFlags(solution.is_center);
    std::vector<std::vector<int>> orders;
    AppendUniqueOrder(orders, BuildBackboneOrder(support, solution.is_center));
    AppendUniqueOrder(orders, BuildCenterMstOrder(support, centers));
    AppendUniqueOrder(orders, centers);
    const int order_limit = std::min(3, std::max(1, SerrEnvInt("FAR_REALIZER_ORDER_LIMIT", 1)));
    if (static_cast<int>(orders.size()) > order_limit) orders.resize(static_cast<size_t>(order_limit));

    for (const auto& base_order : orders) {
        std::vector<ConnectorVariant> variants = {ConnectorVariant::Rsa, ConnectorVariant::EsMst};
        int variant_mode = SerrEnvInt("FAR_REALIZER_VARIANT_MODE", 3);
        if (FarHighDegreeFastMode(net)) {
            variant_mode = SerrEnvInt("FAR_HD_FAST_REALIZER_VARIANT_MODE", 1);
        }
        if (variant_mode == 2) variants = {ConnectorVariant::Rsa};
        else if (variant_mode == 3) variants = {ConnectorVariant::EsMst};
        auto update_best = [&](CandidateResult& result, Tree* tree_to_save) {
            if (!BetterCandidate(result, best)) return;
            best.valid = result.valid;
            best.wl = result.wl;
            best.max_stretch = result.max_stretch;
            if (tree_to_save) {
                CopyTreeForNet(best.tree, *tree_to_save, net);
            } else {
                MoveTree(best.tree, result.tree);
            }
        };
        const auto center_net = MakeCenterNet(support, base_order);
        for (ConnectorVariant variant : variants) {
            const int refine_limit = std::min(2, std::max(1, SerrEnvInt("FAR_REALIZER_REFINE", 2)));
            Tree candidate;
            BuildRealizedTreeFromCenterNet(net, support, solution.block_center, solution.is_center, center_net, variant, eps, candidate);
            if (refine_limit > 1) {
                bool eval_raw = SerrEnvInt("FAR_REALIZER_EVAL_RAW", 1) != 0;
                const int eval_raw_min_centers = SerrEnvInt("FAR_REALIZER_EVAL_RAW_MIN_CENTERS", -1);
                if (eval_raw && eval_raw_min_centers >= 0 &&
                    static_cast<int>(centers.size()) < eval_raw_min_centers) {
                    eval_raw = false;
                }
                CandidateResult raw_result;
                EvaluateTreeMetricsInPlace(candidate, net, eps, raw_result);
                if (eval_raw) update_best(raw_result, &candidate);
                if (!raw_result.valid) continue;
                ApplyFullRefine(candidate, eps);
            }
            CandidateResult result;
            EvaluateTree(candidate, net, eps, result);
            update_best(result, nullptr);
        }
    }
}


struct FarRawInterval {
    int pin_id = -1;
    int sink_old = -1;
    bool root_covered = false;
    int frontier_old = -1;
    int edge_child = -1;
    DTYPE edge_offset = 0;
};

struct FarInterval {
    int pin_id = -1;
    int sink = -1;
    int frontier = -1;
    bool root_covered = false;
    int witness = -1;
};

struct FarRepairAttempt {
    CandidateResult result;
    FarSaltBuilder::Stats stats;
};

Point PointOnL1Edge(const Point& parent, const Point& child, DTYPE offset, bool y_first) {
    const DTYPE dx = std::abs(child.x - parent.x);
    const DTYPE dy = std::abs(child.y - parent.y);
    Point out = parent;

    if (!y_first) {
        if (offset <= dx) {
            out.x = (child.x >= parent.x) ? (parent.x + offset) : (parent.x - offset);
            return out;
        }
        out.x = child.x;
        const DTYPE rem = offset - dx;
        out.y = (child.y >= parent.y) ? (parent.y + std::min(rem, dy)) : (parent.y - std::min(rem, dy));
        return out;
    }

    if (offset <= dy) {
        out.y = (child.y >= parent.y) ? (parent.y + offset) : (parent.y - offset);
        return out;
    }
    out.y = child.y;
    const DTYPE rem = offset - dy;
    out.x = (child.x >= parent.x) ? (parent.x + std::min(rem, dx)) : (parent.x - std::min(rem, dx));
    return out;
}

double PsiOnEdgeOffset(
    const SupportTree& support,
    int edge_child,
    DTYPE offset,
    DTYPE suffix_at_child,
    const Point& root_loc,
    bool y_first
) {
    const int parent = support.parent[static_cast<size_t>(edge_child)];
    const Point x = PointOnL1Edge(support.loc[static_cast<size_t>(parent)], support.loc[static_cast<size_t>(edge_child)], offset, y_first);
    const DTYPE w = support.edge_w[static_cast<size_t>(edge_child)];
    return static_cast<double>(EdgeWeight(root_loc, x)) + static_cast<double>(suffix_at_child + (w - offset));
}

double SupportMaxStretch(const SupportTree& support, const Net& net) {
    double max_stretch = 1.0;
    for (const auto& pin : net.pins) {
        if (!pin || pin->IsSource()) continue;
        const int node = support.pin_to_node[static_cast<size_t>(pin->id)];
        if (node < 0) continue;
        const DTYPE rho = EdgeWeight(support.loc[static_cast<size_t>(support.root)], pin->loc);
        if (rho <= 0) continue;
        const double stretch = static_cast<double>(support.root_dist[static_cast<size_t>(node)]) / static_cast<double>(rho);
        if (stretch > max_stretch) max_stretch = stretch;
    }
    return max_stretch;
}

bool ComputeFarRawIntervals(
    const Net& net,
    const SupportTree& support,
    double eps,
    std::vector<FarRawInterval>& intervals,
    std::vector<std::vector<DTYPE>>& split_offsets,
    FarSaltBuilder::Stats& stats,
    bool y_first
) {
    intervals.clear();
    split_offsets.assign(static_cast<size_t>(support.n), {});
    const double alpha = 1.0 + eps;
    const Point root_loc = support.loc[static_cast<size_t>(support.root)];

    for (const auto& pin : net.pins) {
        if (!pin || pin->IsSource()) continue;
        if (pin->id < 0 || pin->id >= static_cast<int>(support.pin_to_node.size())) return false;
        const int sink = support.pin_to_node[static_cast<size_t>(pin->id)];
        if (sink < 0) return false;
        const DTYPE rho_sink = EdgeWeight(root_loc, pin->loc);
        if (rho_sink <= 0) continue;

        FarRawInterval interval;
        interval.pin_id = pin->id;
        interval.sink_old = sink;
        const double target = alpha * static_cast<double>(rho_sink);
        const double root_psi = static_cast<double>(support.root_dist[static_cast<size_t>(sink)]);
        if (root_psi <= target + kTol) {
            interval.root_covered = true;
            interval.frontier_old = support.root;
            ++stats.root_covered_count;
            intervals.push_back(interval);
            continue;
        }

        ++stats.critical_count;
        int cur = sink;
        DTYPE suffix = 0;
        bool found = false;
        while (cur != support.root) {
            const int parent = support.parent[static_cast<size_t>(cur)];
            if (parent < 0) return false;
            const DTYPE w = support.edge_w[static_cast<size_t>(cur)];
            const double psi_child = static_cast<double>(EdgeWeight(root_loc, support.loc[static_cast<size_t>(cur)])) +
                                     static_cast<double>(suffix);
            const double psi_parent = static_cast<double>(EdgeWeight(root_loc, support.loc[static_cast<size_t>(parent)])) +
                                      static_cast<double>(suffix + w);
            if (psi_parent > target + kTol && psi_child <= target + kTol) {
                DTYPE lo = 0;
                DTYPE hi = w;
                while (lo < hi) {
                    const DTYPE mid = lo + (hi - lo) / 2;
                    if (PsiOnEdgeOffset(support, cur, mid, suffix, root_loc, y_first) <= target + kTol) {
                        hi = mid;
                    } else {
                        lo = mid + 1;
                    }
                }
                interval.edge_child = cur;
                interval.edge_offset = lo;
                if (lo <= 0) {
                    interval.frontier_old = parent;
                    interval.edge_child = -1;
                } else if (lo >= w) {
                    interval.frontier_old = cur;
                    interval.edge_child = -1;
                } else {
                    split_offsets[static_cast<size_t>(cur)].push_back(lo);
                }
                found = true;
                break;
            }
            suffix += w;
            cur = parent;
        }

        if (!found) {
            // Numerical fallback: the sink itself is always feasible for eps >= 0.
            interval.frontier_old = sink;
            interval.edge_child = -1;
        }
        intervals.push_back(interval);
    }

    for (auto& offsets : split_offsets) {
        std::sort(offsets.begin(), offsets.end());
        offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
    }
    return true;
}

bool BuildSplitFarSupport(
    const SupportTree& support,
    const std::vector<std::vector<DTYPE>>& split_offsets,
    const std::vector<FarRawInterval>& raw_intervals,
    SupportTree& out,
    std::vector<FarInterval>& intervals,
    bool y_first
) {
    size_t split_count = 0;
    for (const auto& offsets : split_offsets) split_count += offsets.size();
    std::vector<Point> loc = support.loc;
    loc.reserve(static_cast<size_t>(support.n) + split_count);
    std::vector<std::shared_ptr<Pin>> pin_ref = support.pin_ref;
    pin_ref.reserve(static_cast<size_t>(support.n) + split_count);
    std::vector<int> old_to_new(static_cast<size_t>(support.n), -1);
    for (int v = 0; v < support.n; ++v) old_to_new[static_cast<size_t>(v)] = v;

    std::vector<int> parent(static_cast<size_t>(support.n), -1);
    parent.reserve(static_cast<size_t>(support.n) + split_count);
    std::vector<std::vector<std::pair<DTYPE, int>>> split_id(static_cast<size_t>(support.n));

    for (int v = 0; v < support.n; ++v) {
        const int p = support.parent[static_cast<size_t>(v)];
        if (p < 0) continue;
        int prev = old_to_new[static_cast<size_t>(p)];
        for (DTYPE offset : split_offsets[static_cast<size_t>(v)]) {
            const int id = static_cast<int>(loc.size());
            loc.push_back(PointOnL1Edge(support.loc[static_cast<size_t>(p)], support.loc[static_cast<size_t>(v)], offset, y_first));
            pin_ref.push_back(nullptr);
            parent.push_back(prev);
            split_id[static_cast<size_t>(v)].push_back({offset, id});
            prev = id;
        }
        parent[static_cast<size_t>(old_to_new[static_cast<size_t>(v)])] = prev;
    }

    std::vector<int> pin_to_node = support.pin_to_node;
    for (size_t pin_id = 0; pin_id < pin_to_node.size(); ++pin_id) {
        const int old_node = pin_to_node[pin_id];
        if (old_node >= 0) pin_to_node[pin_id] = old_to_new[static_cast<size_t>(old_node)];
    }

    auto frontier_new_id = [&](const FarRawInterval& raw) -> int {
        if (raw.edge_child < 0) {
            if (raw.frontier_old < 0) return -1;
            return old_to_new[static_cast<size_t>(raw.frontier_old)];
        }
        for (const auto& entry : split_id[static_cast<size_t>(raw.edge_child)]) {
            if (entry.first == raw.edge_offset) return entry.second;
        }
        return -1;
    };

    intervals.clear();
    intervals.reserve(raw_intervals.size());
    for (const auto& raw : raw_intervals) {
        if (raw.pin_id < 0 || raw.pin_id >= static_cast<int>(pin_to_node.size())) return false;
        FarInterval interval;
        interval.pin_id = raw.pin_id;
        interval.sink = pin_to_node[static_cast<size_t>(raw.pin_id)];
        interval.frontier = frontier_new_id(raw);
        interval.root_covered = raw.root_covered;
        if (interval.sink < 0 || interval.frontier < 0) return false;
        intervals.push_back(interval);
    }

    out = BuildSupportTreeFromArrays(loc, pin_ref, pin_to_node, parent, support.name + ":FAR_SPLIT", support.param);
    return true;
}

bool NodeOnAncestorPath(const SupportTree& tree, int top, int bottom, int node) {
    return tree.InSubtree(top, node) && tree.InSubtree(node, bottom);
}

bool AssignFarWitnesses(const SupportTree& support, const std::vector<int>& anchors, std::vector<FarInterval>& intervals) {
    for (auto& interval : intervals) {
        int best = -1;
        DTYPE best_depth = -1;
        for (int anchor : anchors) {
            if (!NodeOnAncestorPath(support, interval.frontier, interval.sink, anchor)) continue;
            const DTYPE depth = support.root_dist[static_cast<size_t>(anchor)];
            if (best < 0 || depth > best_depth || (depth == best_depth && anchor < best)) {
                best = anchor;
                best_depth = depth;
            }
        }
        if (best < 0) return false;
        interval.witness = best;
    }
    return true;
}

bool SelectFarAnchorsAndWitnesses(
    const SupportTree& support,
    std::vector<FarInterval>& intervals,
    std::vector<unsigned char>& is_anchor,
    std::vector<int>& anchors
) {
    is_anchor.assign(static_cast<size_t>(support.n), 0);
    anchors.clear();

    std::vector<int> demands;
    demands.reserve(intervals.size());
    for (int i = 0; i < static_cast<int>(intervals.size()); ++i) {
        if (!intervals[static_cast<size_t>(i)].root_covered) demands.push_back(i);
    }
    std::sort(demands.begin(), demands.end(), [&](int lhs, int rhs) {
        const int lf = intervals[static_cast<size_t>(lhs)].frontier;
        const int rf = intervals[static_cast<size_t>(rhs)].frontier;
        const DTYPE ld = support.root_dist[static_cast<size_t>(lf)];
        const DTYPE rd = support.root_dist[static_cast<size_t>(rf)];
        if (ld != rd) return ld > rd;
        return lf < rf;
    });

    auto add_anchor = [&](int node) {
        if (!is_anchor[static_cast<size_t>(node)]) {
            is_anchor[static_cast<size_t>(node)] = 1;
            anchors.push_back(node);
        }
    };

    for (int idx : demands) {
        const auto& interval = intervals[static_cast<size_t>(idx)];
        bool covered = false;
        for (int anchor : anchors) {
            if (NodeOnAncestorPath(support, interval.frontier, interval.sink, anchor)) {
                covered = true;
                break;
            }
        }
        if (!covered) add_anchor(interval.frontier);
    }
    add_anchor(support.root);
    return AssignFarWitnesses(support, anchors, intervals);
}

bool BuildFarPartition(
    const SupportTree& support,
    const std::vector<FarInterval>& intervals,
    const std::vector<unsigned char>& is_anchor,
    DpSolution& solution
) {
    solution = DpSolution{};
    solution.block_center.assign(static_cast<size_t>(support.n), -1);
    solution.is_center = is_anchor;

    for (int v = 0; v < support.n; ++v) {
        if (solution.is_center[static_cast<size_t>(v)]) solution.block_center[static_cast<size_t>(v)] = v;
    }
    if (!solution.is_center[static_cast<size_t>(support.root)]) return false;

    for (const auto& interval : intervals) {
        int cur = interval.sink;
        const int anchor = interval.witness;
        if (anchor < 0) return false;
        while (true) {
            const int existing = solution.block_center[static_cast<size_t>(cur)];
            if (existing >= 0 && existing != anchor) return false;
            solution.block_center[static_cast<size_t>(cur)] = anchor;
            if (cur == anchor) break;
            cur = support.parent[static_cast<size_t>(cur)];
            if (cur < 0) return false;
        }
    }
    return true;
}

std::vector<int> FarRetainedCounts(const SupportTree& support, const std::vector<FarInterval>& intervals) {
    std::vector<int> retained_count(static_cast<size_t>(support.n), 0);
    for (const auto& interval : intervals) {
        const int witness = interval.witness;
        if (witness < 0) continue;
        int cur = interval.sink;
        while (cur >= 0 && cur != witness) {
            ++retained_count[static_cast<size_t>(cur)];
            cur = support.parent[static_cast<size_t>(cur)];
        }
    }
    return retained_count;
}

void BuildFarFastRealizedCandidate(
    const Net& net,
    const SupportTree& support,
    const DpSolution& solution,
    double eps,
    CandidateResult& best,
    const FarFastRealizerOptions* options = nullptr
) {
    FarProfileScope profile_total("realizer.fast_total");
    best = CandidateResult{};
    const auto order = BuildBackboneOrder(support, solution.is_center);
    const int refine_count =
        options && options->refine_count >= 0
            ? std::min(2, std::max(1, options->refine_count))
            : std::min(2, std::max(1, SerrEnvInt("FAR_FAST_REFINE_COUNT", 2)));
    const int variant_mode =
        options && options->variant_mode >= 0
            ? options->variant_mode
            : SerrEnvInt("FAR_FAST_VARIANT_MODE", 0);
    const int full_mask = SerrEnvInt("FAR_FULL_REFINE_MASK", 31);
    const int fast_mask = SerrEnvInt("FAR_FAST_REFINE_MASK", full_mask);
    const int refine_mask =
        options && options->refine_mask >= 0 ? options->refine_mask : fast_mask;
    const int full_substitute_max_rounds = std::max(0, SerrEnvInt("FAR_SUBSTITUTE_MAX_ROUNDS", 0));
    const int fast_substitute_max_rounds =
        std::max(0, SerrEnvInt("FAR_FAST_SUBSTITUTE_MAX_ROUNDS", full_substitute_max_rounds));
    int substitute_max_rounds =
        options && options->substitute_max_rounds >= 0
            ? std::max(0, options->substitute_max_rounds)
            : fast_substitute_max_rounds;
    if ((!options || options->substitute_max_rounds < 0) && FarHighDegreeFastMode(net)) {
        substitute_max_rounds = std::max(1, SerrEnvInt("FAR_HD_FAST_SUBSTITUTE_MAX_ROUNDS", 1));
    }
    const int substitute_use_rtree =
        options && options->substitute_use_rtree >= 0
            ? options->substitute_use_rtree
            : SerrEnvInt("FAR_FAST_SUBSTITUTE_USE_RTREE", -1);
    const int substitute_linear_candidate_cap =
        options && options->substitute_linear_candidate_cap > 0
            ? std::max(0, options->substitute_linear_candidate_cap)
            : std::max(0, SerrEnvInt("FAR_FAST_SUBSTITUTE_LINEAR_CANDIDATE_CAP", 0));
    const int substitute_linear_query_mode =
        options && options->substitute_linear_query_mode > 0
            ? std::max(0, options->substitute_linear_query_mode)
            : std::max(0, SerrEnvInt("FAR_FAST_SUBSTITUTE_LINEAR_QUERY_MODE", 0));
    std::vector<ConnectorVariant> variants;
    if (variant_mode == 1) variants = {ConnectorVariant::Rsa};
    else if (variant_mode == 2) variants = {ConnectorVariant::EsMst};
    else variants = {ConnectorVariant::Rsa, ConnectorVariant::EsMst};
    auto update_best = [&](CandidateResult& result, Tree* tree_to_save) {
        if (!BetterCandidate(result, best)) return;
        best.valid = result.valid;
        best.wl = result.wl;
        best.max_stretch = result.max_stretch;
        if (tree_to_save) {
            CopyTreeForNet(best.tree, *tree_to_save, net);
        } else {
            MoveTree(best.tree, result.tree);
        }
    };
    const auto center_net = MakeCenterNet(support, order);
    for (ConnectorVariant variant : variants) {
        Tree candidate;
        {
            FarProfileScope profile("realizer.fast_build_tree");
            BuildRealizedTreeFromCenterNet(net, support, solution.block_center, solution.is_center, center_net, variant, eps, candidate);
        }
        if (refine_count > 1) {
            CandidateResult raw_result;
            EvaluateTreeMetricsInPlace(candidate, net, eps, raw_result);
            update_best(raw_result, &candidate);
            if (!raw_result.valid) continue;
            {
                FarProfileScope profile("realizer.fast_refine");
                ApplyRefineMask(candidate, eps, refine_mask, substitute_max_rounds, substitute_use_rtree,
                                substitute_linear_candidate_cap,
                                substitute_linear_query_mode);
            }
        }
        CandidateResult result;
        EvaluateTree(candidate, net, eps, result);
        update_best(result, nullptr);
    }
}

// Historical graph/LARS/quotient/SERR candidate families were removed from the
// minimal SOTA package. The retained solver path is the FLUTE support repair
// flow below plus the bounded SUPER tail pass in RunImpl.

FarRepairAttempt BuildFarRepairAttempt(const Net& net, const SupportTree& base, double eps, bool full_mode, bool y_first) {
    FarProfileContextScope profile_ctx(full_mode ? "attempt.full" : "attempt.fast");
    FarProfileScope profile_total("attempt.total");
    FarRepairAttempt attempt;
    attempt.stats.base_name = base.name + (y_first ? ":YX" : ":XY");
    attempt.stats.base_wl = base.total_w;
    attempt.stats.base_stretch = SupportMaxStretch(base, net);

    std::vector<FarRawInterval> raw_intervals;
    std::vector<std::vector<DTYPE>> split_offsets;
    {
        FarProfileScope profile("attempt.raw_intervals");
        if (!ComputeFarRawIntervals(net, base, eps, raw_intervals, split_offsets, attempt.stats, y_first)) return attempt;
    }

    SupportTree split_support;
    std::vector<FarInterval> intervals;
    {
        FarProfileScope profile("attempt.split_support");
        if (!BuildSplitFarSupport(base, split_offsets, raw_intervals, split_support, intervals, y_first)) return attempt;
    }

    std::vector<unsigned char> is_anchor;
    std::vector<int> anchors;
    {
        FarProfileScope profile("attempt.select_base_anchors");
        if (!SelectFarAnchorsAndWitnesses(split_support, intervals, is_anchor, anchors)) return attempt;
    }

    auto build_with_anchors = [&](const std::vector<unsigned char>& anchor_flags,
                                  const std::vector<int>& anchor_list,
                                  bool use_full_realizer,
                                  CandidateResult& out,
                                  const FarFastRealizerOptions* fast_options = nullptr) -> bool {
        std::vector<FarInterval> assigned = intervals;
        {
            FarProfileScope profile("build.assign_witnesses");
            if (!AssignFarWitnesses(split_support, anchor_list, assigned)) return false;
        }

        DpSolution solution;
        {
            FarProfileScope profile("build.partition");
            if (!BuildFarPartition(split_support, assigned, anchor_flags, solution)) return false;
        }

        if (use_full_realizer) {
            FarProfileScope profile("build.full_realizer");
            BuildBestRealizedCandidate(net, split_support, solution, eps, out);
        } else {
            FarProfileScope profile("build.fast_realizer");
            BuildFarFastRealizedCandidate(net, split_support, solution, eps, out, fast_options);
        }
        return out.valid;
    };

    auto retained_delta_with_counts = [&](int node,
                                          const std::vector<FarInterval>& assigned,
                                          const std::vector<int>& retained_count) -> DTYPE {
        if (node < 0 || node >= split_support.n) return static_cast<DTYPE>(0);
        std::vector<int> affected_edge_count(static_cast<size_t>(split_support.n), 0);
        std::vector<int> affected_touched;
        int affected = 0;
        for (const auto& interval : assigned) {
            if (!NodeOnAncestorPath(split_support, interval.frontier, interval.sink, node)) continue;
            const int witness = interval.witness;
            if (witness < 0) continue;
            if (split_support.root_dist[static_cast<size_t>(node)] <=
                split_support.root_dist[static_cast<size_t>(witness)]) {
                continue;
            }
            ++affected;
            int cur = node;
            while (cur >= 0 && cur != witness) {
                int& count = affected_edge_count[static_cast<size_t>(cur)];
                if (count == 0) affected_touched.push_back(cur);
                ++count;
                cur = split_support.parent[static_cast<size_t>(cur)];
            }
        }
        if (affected == 0) return static_cast<DTYPE>(0);

        DTYPE delta = 0;
        for (int edge : affected_touched) {
            if (affected_edge_count[static_cast<size_t>(edge)] == retained_count[static_cast<size_t>(edge)]) {
                delta += split_support.edge_w[static_cast<size_t>(edge)];
            }
        }
        return delta;
    };

    auto collect_anchor_pool = [&]() {
        std::vector<unsigned char> pool_flag(static_cast<size_t>(split_support.n), 0);
        std::vector<int> pool;
        pool.reserve(intervals.size());
        for (const auto& interval : intervals) {
            int cur = interval.sink;
            while (cur >= 0) {
                if (!pool_flag[static_cast<size_t>(cur)]) {
                    pool_flag[static_cast<size_t>(cur)] = 1;
                    pool.push_back(cur);
                }
                if (cur == interval.frontier) break;
                cur = split_support.parent[static_cast<size_t>(cur)];
            }
        }
        return pool;
    };

    auto rank_pool_by_retained_release = [&](std::vector<int>& pool,
                                             const std::vector<unsigned char>& anchor_flags,
                                             const std::vector<FarInterval>& assigned) {
        const auto retained_count = FarRetainedCounts(split_support, assigned);
        std::vector<DTYPE> release(static_cast<size_t>(split_support.n), 0);
        std::vector<int> affected_count(static_cast<size_t>(split_support.n), 0);
        for (int node : pool) {
            if (node < 0 || node >= split_support.n || anchor_flags[static_cast<size_t>(node)]) continue;
            release[static_cast<size_t>(node)] = retained_delta_with_counts(node, assigned, retained_count);
            for (const auto& interval : assigned) {
                if (!NodeOnAncestorPath(split_support, interval.frontier, interval.sink, node)) continue;
                const int witness = interval.witness;
                if (witness < 0) continue;
                if (split_support.root_dist[static_cast<size_t>(node)] <=
                    split_support.root_dist[static_cast<size_t>(witness)]) {
                    continue;
                }
                ++affected_count[static_cast<size_t>(node)];
            }
        }
        std::sort(pool.begin(), pool.end(), [&](int lhs, int rhs) {
            const DTYPE lr = release[static_cast<size_t>(lhs)];
            const DTYPE rr = release[static_cast<size_t>(rhs)];
            if (lr != rr) return lr > rr;
            const int la = affected_count[static_cast<size_t>(lhs)];
            const int ra = affected_count[static_cast<size_t>(rhs)];
            if (la != ra) return la > ra;
            const DTYPE ld = split_support.root_dist[static_cast<size_t>(lhs)];
            const DTYPE rd = split_support.root_dist[static_cast<size_t>(rhs)];
            if (ld != rd) return ld > rd;
            return lhs < rhs;
        });
    };

    auto update_stats = [&]() {
        if (attempt.result.valid) {
            attempt.stats.candidate_wl = attempt.result.wl;
            attempt.stats.candidate_stretch = attempt.result.max_stretch;
            attempt.stats.repair_cost = attempt.result.wl - attempt.stats.base_wl;
        }
    };

    {
        FarProfileContextScope profile_build_ctx("initial");
        FarProfileScope profile("attempt.build_initial");
        build_with_anchors(is_anchor, anchors, full_mode, attempt.result);
    }
    int final_anchor_count = static_cast<int>(anchors.size());

    std::vector<int> pool;
    {
        FarProfileScope profile("attempt.pool_collect_rank");
        pool = collect_anchor_pool();
        std::vector<FarInterval> assigned = intervals;
        if (AssignFarWitnesses(split_support, anchors, assigned)) {
            rank_pool_by_retained_release(pool, is_anchor, assigned);
        } else {
            std::sort(pool.begin(), pool.end(), [&](int lhs, int rhs) {
                const DTYPE ld = split_support.root_dist[static_cast<size_t>(lhs)];
                const DTYPE rd = split_support.root_dist[static_cast<size_t>(rhs)];
                if (ld != rd) return ld > rd;
                return lhs < rhs;
            });
        }
        const int pool_limit = std::min(static_cast<int>(pool.size()), std::max(0, SerrEnvInt("FAR_ANCHOR_SCAN_POOL", 7)));
        if (pool_limit > 0 && static_cast<int>(pool.size()) > pool_limit) pool.resize(static_cast<size_t>(pool_limit));
    }

    int direct_count = std::min(8, std::max(0, SerrEnvInt("FAR_ANCHOR_DIRECT_COUNT", 4)));
    if (FarHighDegreeFastMode(net)) {
        direct_count = std::min(8, std::max(0, SerrEnvInt("FAR_HD_FAST_ANCHOR_DIRECT_COUNT", 0)));
    }
    if (direct_count <= 0) {
        const int hd_direct_count = std::min(8, std::max(0, SerrEnvInt("FAR_HD_DIRECT_COUNT", 0)));
        const int hd_min_degree = std::max(1, SerrEnvInt("FAR_HD_DIRECT_MIN_DEG", 30));
        const int hd_anchor_limit = SerrEnvInt("FAR_HD_DIRECT_IF_ANCHORS_LE", -1);
        const int hd_critical_min = SerrEnvInt("FAR_HD_DIRECT_IF_CRITICAL_GE", -1);
        const bool degree_ok = static_cast<int>(net.pins.size()) >= hd_min_degree;
        const bool has_online_gate = hd_anchor_limit >= 0 || hd_critical_min >= 0;
        const bool anchor_ok = hd_anchor_limit < 0 || static_cast<int>(anchors.size()) <= hd_anchor_limit;
        const bool critical_ok = hd_critical_min < 0 || attempt.stats.critical_count >= hd_critical_min;
        if (hd_direct_count > 0 && degree_ok && has_online_gate && anchor_ok && critical_ok) {
            direct_count = hd_direct_count;
        }
    }
    if (direct_count > 0) {
        FarProfileScope profile_direct("attempt.direct_total");
        std::vector<unsigned char> direct_flags = is_anchor;
        std::vector<int> direct_anchors = anchors;
        std::vector<int> direct_added_nodes;
        for (int node : pool) {
            if (node < 0 || node >= split_support.n) continue;
            if (direct_flags[static_cast<size_t>(node)]) continue;
            direct_flags[static_cast<size_t>(node)] = 1;
            direct_anchors.push_back(node);
            direct_added_nodes.push_back(node);
            if (static_cast<int>(direct_added_nodes.size()) >= direct_count) break;
        }
        if (!direct_added_nodes.empty()) {
            CandidateResult direct_result;
            const bool direct_full_realizer = SerrEnvInt("FAR_ANCHOR_DIRECT_FAST", 0) == 0;
            {
                FarProfileContextScope profile_ctx("direct");
                build_with_anchors(direct_flags, direct_anchors, direct_full_realizer, direct_result);
            }
            if (BetterCandidate(direct_result, attempt.result)) {
                attempt.result.valid = direct_result.valid;
                attempt.result.wl = direct_result.wl;
                attempt.result.max_stretch = direct_result.max_stretch;
                MoveTree(attempt.result.tree, direct_result.tree);
                final_anchor_count = static_cast<int>(direct_anchors.size());
            }
        }
    }

    const int scan_iters = std::min(4, std::max(0, SerrEnvInt("FAR_ANCHOR_SCAN_ITERS", 2)));
    if (scan_iters > 0) {
        FarProfileScope profile_scan_total("attempt.scan_total");
        std::vector<unsigned char> cur_flags = is_anchor;
        std::vector<int> cur_anchors = anchors;
        CandidateResult scan_current;
        {
            FarProfileContextScope profile_ctx("scan_seed");
            build_with_anchors(cur_flags, cur_anchors, false, scan_current);
        }

        const int continue_gain_ppm = SerrEnvInt("FAR_ANCHOR_CONTINUE_MIN_GAIN_PPM", 0);
        const int continue_rescue_critical = std::max(0, SerrEnvInt("FAR_ANCHOR_CONTINUE_RESCUE_MIN_CRITICAL", 0));
        const int continue_rescue_covered = SerrEnvInt("FAR_ANCHOR_CONTINUE_RESCUE_MAX_COVERED", -1);
        const bool rescue_continue =
            (continue_rescue_critical > 0 && attempt.stats.critical_count >= continue_rescue_critical) ||
            (continue_rescue_covered >= 0 && attempt.stats.root_covered_count <= continue_rescue_covered);

        for (int iter = 0; iter < scan_iters; ++iter) {
            const DTYPE scan_before_wl = scan_current.valid ? scan_current.wl : std::numeric_limits<DTYPE>::max();
            int best_node = -1;
            CandidateResult best_scan = scan_current;

            struct ScanStructRank {
                double score = -kInf;
                DTYPE release = 0;
                DTYPE depth_gain = 0;
                DTYPE nearest_anchor = 0;
                int affected = 0;
                int old_witness_count = 0;
                int rank = -1;
            };
            std::vector<ScanStructRank> struct_ranked;

            auto add_unique_rank = [&](std::vector<int>& ranks, int rank) {
                if (rank < 0 || rank >= static_cast<int>(pool.size())) return;
                if (std::find(ranks.begin(), ranks.end(), rank) != ranks.end()) return;
                ranks.push_back(rank);
            };

            auto evaluate_scan_rank = [&](int rank) -> bool {
                if (rank < 0 || rank >= static_cast<int>(pool.size())) return false;
                FarProfileContextScope profile_ctx("scan_trial");
                FarProfileScope profile("attempt.scan_eval_rank");
                const int node = pool[static_cast<size_t>(rank)];
                if (node < 0 || node >= split_support.n || cur_flags[static_cast<size_t>(node)]) return false;
                std::vector<unsigned char> trial_flags = cur_flags;
                std::vector<int> trial_anchors = cur_anchors;
                trial_flags[static_cast<size_t>(node)] = 1;
                trial_anchors.push_back(node);
                CandidateResult trial;
                if (!build_with_anchors(trial_flags, trial_anchors, false, trial)) return false;
                const bool improved = BetterCandidate(trial, scan_current);
                if (BetterCandidate(trial, best_scan)) {
                    best_scan.valid = trial.valid;
                    best_scan.wl = trial.wl;
                    best_scan.max_stretch = trial.max_stretch;
                    MoveTree(best_scan.tree, trial.tree);
                    best_node = node;
                }
                return improved;
            };
            {
                FarProfileContextScope profile_ctx("scan_struct");
                FarProfileScope profile("attempt.scan_struct_total");
                std::vector<FarInterval> struct_assigned = intervals;
                if (AssignFarWitnesses(split_support, cur_anchors, struct_assigned)) {
                    const auto retained_count = FarRetainedCounts(split_support, struct_assigned);
                    const double release_weight =
                        static_cast<double>(SerrEnvInt("FAR_ANCHOR_SCAN_STRUCT_RELEASE_PPM", 1000000)) / 1000000.0;
                    const double depth_weight =
                        static_cast<double>(SerrEnvInt("FAR_ANCHOR_SCAN_STRUCT_DEPTH_PPM", 20000)) / 1000000.0;
                    const double conn_weight =
                        static_cast<double>(SerrEnvInt("FAR_ANCHOR_SCAN_STRUCT_CONN_PPM", 100000)) / 1000000.0;
                    const double affected_weight =
                        static_cast<double>(SerrEnvInt("FAR_ANCHOR_SCAN_STRUCT_AFFECTED_PPM", 0)) / 1000000.0;
                    const double old_witness_weight =
                        static_cast<double>(SerrEnvInt("FAR_ANCHOR_SCAN_STRUCT_OLD_WITNESS_PPM", 0)) / 1000000.0;
                    struct_ranked.reserve(pool.size());
                    for (int rank = 0; rank < static_cast<int>(pool.size()); ++rank) {
                        const int node = pool[static_cast<size_t>(rank)];
                        if (node < 0 || node >= split_support.n || cur_flags[static_cast<size_t>(node)]) continue;

                        ScanStructRank item;
                        item.rank = rank;
                        std::vector<int> affected_edge_count(static_cast<size_t>(split_support.n), 0);
                        std::vector<int> affected_touched;
                        std::vector<int> old_witness_count(static_cast<size_t>(split_support.n), 0);
                        std::vector<int> touched_witnesses;
                        for (const auto& interval : struct_assigned) {
                            if (!NodeOnAncestorPath(split_support, interval.frontier, interval.sink, node)) continue;
                            const int witness = interval.witness;
                            if (witness < 0) continue;
                            if (split_support.root_dist[static_cast<size_t>(node)] <=
                                split_support.root_dist[static_cast<size_t>(witness)]) {
                                continue;
                            }
                            ++item.affected;
                            item.depth_gain += split_support.root_dist[static_cast<size_t>(node)] -
                                                split_support.root_dist[static_cast<size_t>(witness)];
                            int& witness_count = old_witness_count[static_cast<size_t>(witness)];
                            if (witness_count == 0) touched_witnesses.push_back(witness);
                            ++witness_count;

                            int cur = node;
                            while (cur >= 0 && cur != witness) {
                                int& count = affected_edge_count[static_cast<size_t>(cur)];
                                if (count == 0) affected_touched.push_back(cur);
                                ++count;
                                cur = split_support.parent[static_cast<size_t>(cur)];
                            }
                        }
                        for (int edge : affected_touched) {
                            if (affected_edge_count[static_cast<size_t>(edge)] ==
                                retained_count[static_cast<size_t>(edge)]) {
                                item.release += split_support.edge_w[static_cast<size_t>(edge)];
                            }
                        }
                        item.old_witness_count = static_cast<int>(touched_witnesses.size());
                        DTYPE nearest_anchor = std::numeric_limits<DTYPE>::max();
                        for (int anchor : cur_anchors) {
                            if (anchor < 0 || anchor >= split_support.n) continue;
                            nearest_anchor = std::min(nearest_anchor,
                                                      EdgeWeight(split_support.loc[static_cast<size_t>(node)],
                                                                 split_support.loc[static_cast<size_t>(anchor)]));
                        }
                        item.nearest_anchor =
                            nearest_anchor == std::numeric_limits<DTYPE>::max() ? 0 : nearest_anchor;
                        if (item.affected <= 0) continue;
                        item.score = release_weight * static_cast<double>(item.release) +
                                     depth_weight * static_cast<double>(item.depth_gain) -
                                     conn_weight * static_cast<double>(item.nearest_anchor) +
                                     affected_weight * static_cast<double>(item.affected) +
                                     old_witness_weight * static_cast<double>(item.old_witness_count);
                        struct_ranked.push_back(item);
                    }
                }
            }
            std::sort(struct_ranked.begin(), struct_ranked.end(),
                      [](const ScanStructRank& lhs, const ScanStructRank& rhs) {
                          if (std::abs(lhs.score - rhs.score) > kTol) return lhs.score > rhs.score;
                          if (lhs.release != rhs.release) return lhs.release > rhs.release;
                          if (lhs.depth_gain != rhs.depth_gain) return lhs.depth_gain > rhs.depth_gain;
                          if (lhs.nearest_anchor != rhs.nearest_anchor) return lhs.nearest_anchor < rhs.nearest_anchor;
                          if (lhs.affected != rhs.affected) return lhs.affected > rhs.affected;
                          if (lhs.old_witness_count != rhs.old_witness_count) return lhs.old_witness_count > rhs.old_witness_count;
                          return lhs.rank < rhs.rank;
                      });
            std::vector<int> exact_ranks;
            const int struct_head = std::min(static_cast<int>(pool.size()),
                                             std::max(0, SerrEnvInt("FAR_ANCHOR_SCAN_STRUCT_HEAD", 1)));
            for (int rank = 0; rank < struct_head; ++rank) add_unique_rank(exact_ranks, rank);
            const int struct_topk = std::min(static_cast<int>(pool.size()),
                                             std::max(0, SerrEnvInt("FAR_ANCHOR_SCAN_STRUCT_TOPK", 5)));
            const int keep = std::min(struct_topk, static_cast<int>(struct_ranked.size()));
            const int late_prune_min_rank = std::max(0, SerrEnvInt("FAR_ANCHOR_SCAN_LATE_PRUNE_MIN_RANK", 1000000));
            const int late_prune_min_iter = std::max(0, SerrEnvInt("FAR_ANCHOR_SCAN_LATE_PRUNE_MIN_ITER", 0));
            const DTYPE late_prune_min_release =
                static_cast<DTYPE>(std::max(0, SerrEnvInt("FAR_ANCHOR_SCAN_LATE_PRUNE_MIN_RELEASE", 0)));
            const int late_prune_min_score_ppm =
                std::max(0, SerrEnvInt("FAR_ANCHOR_SCAN_LATE_PRUNE_MIN_SCORE_PPM", 0));
            const int late_prune_min_affected =
                std::max(0, SerrEnvInt("FAR_ANCHOR_SCAN_LATE_PRUNE_MIN_AFFECTED", 0));
            auto scan_struct_allowed = [&](const ScanStructRank& item) {
                if (iter < late_prune_min_iter) return true;
                if (item.rank < late_prune_min_rank) return true;
                if (late_prune_min_release > 0 && item.release + kTol < late_prune_min_release) return false;
                if (late_prune_min_affected > 0 && item.affected < late_prune_min_affected) return false;
                if (late_prune_min_score_ppm > 0 &&
                    scan_before_wl > 0 && scan_before_wl < std::numeric_limits<DTYPE>::max()) {
                    const double score_ppm =
                        1000000.0 * item.score / std::max(1.0, static_cast<double>(scan_before_wl));
                    if (score_ppm + kTol < static_cast<double>(late_prune_min_score_ppm)) return false;
                }
                return true;
            };
            for (int idx = 0; idx < keep; ++idx) {
                const ScanStructRank& item = struct_ranked[static_cast<size_t>(idx)];
                if (scan_struct_allowed(item)) add_unique_rank(exact_ranks, item.rank);
            }
            std::sort(exact_ranks.begin(), exact_ranks.end());

            for (int rank : exact_ranks) {
                evaluate_scan_rank(rank);
            }

            const int struct_fallback_topk = std::min(static_cast<int>(pool.size()),
                                                      std::max(0, SerrEnvInt("FAR_ANCHOR_SCAN_STRUCT_FALLBACK_TOPK", 7)));
            const int fallback_keep = std::min(struct_fallback_topk, static_cast<int>(struct_ranked.size()));
            if (fallback_keep > keep) {
                bool run_fallback = best_node < 0;
                const int struct_fallback_min_gain_ppm =
                    std::max(0, SerrEnvInt("FAR_ANCHOR_SCAN_STRUCT_FALLBACK_MIN_GAIN_PPM", 2000));
                if (!run_fallback && struct_fallback_min_gain_ppm > 0 &&
                    scan_before_wl > 0 && scan_before_wl < std::numeric_limits<DTYPE>::max()) {
                    const double gain_ppm = 1000000.0 * static_cast<double>(scan_before_wl - best_scan.wl) /
                                            std::max(1.0, static_cast<double>(scan_before_wl));
                    if (gain_ppm + kTol < static_cast<double>(struct_fallback_min_gain_ppm)) run_fallback = true;
                }
                if (run_fallback) {
                    std::vector<int> fallback_ranks;
                    for (int idx = keep; idx < fallback_keep; ++idx) {
                        const ScanStructRank& item = struct_ranked[static_cast<size_t>(idx)];
                        if (scan_struct_allowed(item)) add_unique_rank(fallback_ranks, item.rank);
                    }
                    std::sort(fallback_ranks.begin(), fallback_ranks.end());
                    for (int rank : fallback_ranks) {
                        evaluate_scan_rank(rank);
                    }
                }
            }

            if (best_node < 0) break;
            cur_flags[static_cast<size_t>(best_node)] = 1;
            cur_anchors.push_back(best_node);
            scan_current.valid = best_scan.valid;
            scan_current.wl = best_scan.wl;
            scan_current.max_stretch = best_scan.max_stretch;
            MoveTree(scan_current.tree, best_scan.tree);

            if (continue_gain_ppm > 0 && iter == 0 &&
                scan_before_wl > 0 && scan_before_wl < std::numeric_limits<DTYPE>::max()) {
                const double gain_ppm = 1000000.0 * static_cast<double>(scan_before_wl - scan_current.wl) /
                                        std::max(1.0, static_cast<double>(scan_before_wl));
                if (!rescue_continue && gain_ppm + kTol < static_cast<double>(continue_gain_ppm)) break;
            }
        }

        if (cur_anchors.size() > anchors.size()) {
            CandidateResult augmented;
            {
                FarProfileContextScope profile_ctx("scan_augmented");
                FarProfileScope profile("attempt.scan_augmented_build");
                build_with_anchors(cur_flags, cur_anchors, true, augmented);
            }
            if (BetterCandidate(augmented, attempt.result)) {
                attempt.result.valid = augmented.valid;
                attempt.result.wl = augmented.wl;
                attempt.result.max_stretch = augmented.max_stretch;
                MoveTree(attempt.result.tree, augmented.tree);
                final_anchor_count = static_cast<int>(cur_anchors.size());
            }
        }
    }

    attempt.stats.anchor_count = final_anchor_count;
    update_stats();
    return attempt;
}

}  // namespace

void FarSaltBuilder::RunFull(const Net& net, Tree& tree, double eps) { RunImpl(net, tree, eps); }

void FarSaltBuilder::RunImpl(const Net& net, Tree& tree, double eps) {
    constexpr bool full_mode = true;
    FarProfileRunScope profile_run(net, full_mode);
    last_stats_ = Stats{};

    std::vector<SupportTree> bases;
    auto push_unique = [&](SupportTree candidate) {
        for (const auto& existing : bases) {
            if (SupportsEquivalent(existing, candidate)) return;
        }
        bases.push_back(std::move(candidate));
    };

    // Fixed SOTA support family by default: A3/SALT-compatible refined FLUTE
    // plus A9 FLUTE on the high-degree set.  FAR_SUPPORT_MODE is a default-off
    // high-degree ablation hook:
    //   0: default family, 1: refined FLUTE_R1 only, 2: A9-only when legal.
    int support_mode = SerrEnvInt("FAR_SUPPORT_MODE", 0);
    if (FarHighDegreeFastMode(net)) {
        support_mode = SerrEnvInt("FAR_HD_FAST_SUPPORT_MODE", 2);
    }
    if (support_mode == 1) {
        push_unique(BuildFluteSupport(net, true));
    } else if (support_mode == 2 && net.pins.size() >= 31) {
        push_unique(BuildFluteSupportWithAccuracy(net, 9, false));
    } else {
        push_unique(BuildFluteSupport(net, true));
        if (net.pins.size() >= 31) {
            push_unique(BuildFluteSupportWithAccuracy(net, 9, false));
            const int extra_a9_r1_min_degree = SerrEnvInt("FAR_EXTRA_A9_R1_MIN_DEG", 0);
            if (extra_a9_r1_min_degree > 0 &&
                static_cast<int>(net.pins.size()) >= extra_a9_r1_min_degree) {
                push_unique(BuildFluteSupportWithAccuracy(net, 9, true));
            }
        }
    }

    CandidateResult best_realized;
    for (size_t base_idx = 0; base_idx < bases.size(); ++base_idx) {
        FarProfileContextScope profile_base_ctx(base_idx == 0 ? "main_base0" : "main_base1");
        FarRepairAttempt attempt = BuildFarRepairAttempt(net, bases[base_idx], eps, full_mode, false);
        if (BetterCandidate(attempt.result, best_realized)) {
            best_realized.valid = true;
            best_realized.wl = attempt.result.wl;
            best_realized.max_stretch = attempt.result.max_stretch;
            MoveTree(best_realized.tree, attempt.result.tree);
            last_stats_ = attempt.stats;
        }
        const int extra_yx_min_degree = SerrEnvInt("FAR_EXTRA_YX_MIN_DEG", 0);
        if (extra_yx_min_degree > 0 &&
            static_cast<int>(net.pins.size()) >= extra_yx_min_degree) {
            FarProfileContextScope profile_yx_ctx(base_idx == 0 ? "main_base0_yx" : "main_base1_yx");
            FarRepairAttempt yx_attempt = BuildFarRepairAttempt(net, bases[base_idx], eps, full_mode, true);
            if (BetterCandidate(yx_attempt.result, best_realized)) {
                best_realized.valid = true;
                best_realized.wl = yx_attempt.result.wl;
                best_realized.max_stretch = yx_attempt.result.max_stretch;
                MoveTree(best_realized.tree, yx_attempt.result.tree);
                last_stats_ = yx_attempt.stats;
            }
        }
    }

    const int super_tail_rounds = std::max(0, SerrEnvInt("FAR_SUPER_TAIL_ROUNDS", 1));
    if (best_realized.valid && super_tail_rounds > 0) {
        FarProfileScope profile("run.post_super_tail_total");
        const int kSuperTailRounds = super_tail_rounds;
        const int kSuperTailMask = SerrEnvInt("FAR_SUPER_TAIL_MASK", 383);

        CandidateResult super_tail;
        BuildSuperTailCandidate(net, best_realized, eps, kSuperTailRounds, super_tail, kSuperTailMask);

        if (BetterCandidate(super_tail, best_realized)) {
            best_realized.valid = true;
            best_realized.wl = super_tail.wl;
            best_realized.max_stretch = super_tail.max_stretch;
            MoveTree(best_realized.tree, super_tail.tree);
            if (last_stats_.base_name.empty()) last_stats_.base_name = "SUPER_TAIL";
            else last_stats_.base_name = "SUPER_TAIL:" + last_stats_.base_name;
            last_stats_.candidate_wl = best_realized.wl;
            last_stats_.candidate_stretch = best_realized.max_stretch;
        }
    }

    if (!best_realized.valid) {
        tree.Reset();
        return;
    }

    MoveTree(tree, best_realized.tree);
    tree.net = &net;
    printlog("FAR-SALT chose", last_stats_.base_name, "anchors", last_stats_.anchor_count,
             "critical", last_stats_.critical_count, "repair =", last_stats_.repair_cost,
             "wl =", last_stats_.candidate_wl, "maxStretch =", last_stats_.candidate_stretch);
}

}  // namespace salt
