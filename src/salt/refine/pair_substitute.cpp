#include "refine.h"

#include "salt/base/eval.h"
#include "salt/base/flute.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <limits>
#include <set>
#include <string>

namespace salt {

namespace {

struct PairMove {
    DTYPE wireLengthDelta = 0;
    long long weightedPathDelta = std::numeric_limits<long long>::max();
    shared_ptr<TreeNode> bundleRoot;
    shared_ptr<TreeNode> left;
    shared_ptr<TreeNode> right;
    shared_ptr<TreeNode> supportChild;
    Point supportPoint;
    Point branchPoint;
    bool valid = false;
};

struct LocalFluteSolution {
    bool valid = false;
    DTYPE wl = numeric_limits<DTYPE>::max();
    salt::Tree tree;
    vector<DTYPE> sinkPathLengths;
};

struct TripletMove {
    DTYPE wireLengthDelta = 0;
    long long weightedPathDelta = std::numeric_limits<long long>::max();
    shared_ptr<TreeNode> bundleRoot;
    shared_ptr<TreeNode> innerChild;
    vector<shared_ptr<TreeNode>> roots;
    shared_ptr<TreeNode> supportChild;
    Point supportPoint;
    shared_ptr<TreeNode> localSource;
    vector<DTYPE> localPathLengths;
    bool valid = false;
};

struct QuadMove {
    DTYPE wireLengthDelta = 0;
    long long weightedPathDelta = std::numeric_limits<long long>::max();
    shared_ptr<TreeNode> bundleRoot;
    vector<shared_ptr<TreeNode>> innerChildren;
    vector<shared_ptr<TreeNode>> roots;
    shared_ptr<TreeNode> supportChild;
    Point supportPoint;
    shared_ptr<TreeNode> localSource;
    vector<DTYPE> localPathLengths;
    bool valid = false;
};

struct BundleLocalityLimits {
    bool enabled = false;
    DTYPE maxPathWlSpan = -1;
    int minDepthOffset = std::numeric_limits<int>::min();
    int maxDepthOffset = std::numeric_limits<int>::max();
    bool lcaEnabled = false;
    bool lcaEnumerate = false;
    int maxBundleRootUpFromLca = std::numeric_limits<int>::max();
    int maxSupportParentDownFromLca = std::numeric_limits<int>::max();
};

inline DTYPE Clamp(DTYPE value, DTYPE lo, DTYPE hi) {
    return std::max(lo, std::min(value, hi));
}

inline DTYPE Median3(DTYPE a, DTYPE b, DTYPE c) {
    return a + b + c - std::min({a, b, c}) - std::max({a, b, c});
}

inline Point MedianPoint(const Point& a, const Point& b, const Point& c) {
    return {Median3(a.x, b.x, c.x), Median3(a.y, b.y, c.y)};
}

inline DTYPE ThreePointRsmtLength(const Point& a, const Point& b, const Point& s, Point& branchPoint) {
    branchPoint = MedianPoint(a, b, s);
    return Dist(a, branchPoint) + Dist(b, branchPoint) + Dist(s, branchPoint);
}

inline DTYPE AxisSpanWithPoint(DTYPE a, DTYPE b, DTYPE value) {
    return std::max({a, b, value}) - std::min({a, b, value});
}

inline DTYPE MinAxisSpanWithSegment(DTYPE a, DTYPE b, DTYPE lo, DTYPE hi) {
    const DTYPE abLo = std::min(a, b);
    const DTYPE abHi = std::max(a, b);
    if (hi < abLo) return abHi - hi;
    if (lo > abHi) return lo - abLo;
    return abHi - abLo;
}

inline DTYPE ThreePointRsmtMinLengthOnSupportEdge(const shared_ptr<TreeNode>& supportChild,
                                                  const Point& left,
                                                  const Point& right) {
    auto supportParent = supportChild->parent;
    if (supportParent->loc.x == supportChild->loc.x) {
        const DTYPE xSpan = AxisSpanWithPoint(left.x, right.x, supportChild->loc.x);
        const DTYPE lo = std::min(supportParent->loc.y, supportChild->loc.y);
        const DTYPE hi = std::max(supportParent->loc.y, supportChild->loc.y);
        return xSpan + MinAxisSpanWithSegment(left.y, right.y, lo, hi);
    }

    const DTYPE ySpan = AxisSpanWithPoint(left.y, right.y, supportChild->loc.y);
    const DTYPE lo = std::min(supportParent->loc.x, supportChild->loc.x);
    const DTYPE hi = std::max(supportParent->loc.x, supportChild->loc.x);
    return MinAxisSpanWithSegment(left.x, right.x, lo, hi) + ySpan;
}

inline DTYPE FourPointRsmtLength(const Point& rootPoint, const vector<shared_ptr<TreeNode>>& roots) {
    std::array<DTYPE, 4> xs = {rootPoint.x, roots[0]->loc.x, roots[1]->loc.x, roots[2]->loc.x};
    std::array<DTYPE, 4> ys = {rootPoint.y, roots[0]->loc.y, roots[1]->loc.y, roots[2]->loc.y};
    std::sort(xs.begin(), xs.end());
    std::sort(ys.begin(), ys.end());
    return (xs[3] - xs[0]) + (ys[3] - ys[0]) + std::min(xs[2] - xs[1], ys[2] - ys[1]);
}

int RefineEnvInt(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    return (end != value) ? static_cast<int>(parsed) : fallback;
}

int RefineEnvInt(const std::string& name, int fallback) {
    return RefineEnvInt(name.c_str(), fallback);
}

BundleLocalityLimits MakeBundleLocalityLimits(const char* passPrefix) {
    BundleLocalityLimits limits;
    const bool globalEnabled = RefineEnvInt("SALT_BUNDLE_LOCALITY", 0) != 0;
    const bool globalLcaEnabled = RefineEnvInt("SALT_BUNDLE_LCA_LOCALITY", 0) != 0;
    const bool globalLcaEnumerate = RefineEnvInt("SALT_BUNDLE_LCA_ENUMERATE", 0) != 0;
    const std::string prefix = passPrefix ? passPrefix : "";
    limits.enabled = RefineEnvInt(prefix + "_LOCALITY", globalEnabled ? 1 : 0) != 0;
    limits.lcaEnabled = RefineEnvInt(prefix + "_LCA_LOCALITY", globalLcaEnabled ? 1 : 0) != 0;
    limits.lcaEnumerate = RefineEnvInt(prefix + "_LCA_ENUMERATE", globalLcaEnumerate ? 1 : 0) != 0;
    limits.lcaEnabled = limits.lcaEnabled || limits.lcaEnumerate;
    limits.enabled = limits.enabled || limits.lcaEnabled;
    if (!limits.enabled) return limits;

    limits.maxPathWlSpan = std::max(
        -1, RefineEnvInt(prefix + "_LOCALITY_PATH_SPAN",
                         RefineEnvInt("SALT_BUNDLE_LOCALITY_PATH_SPAN", -1)));
    limits.minDepthOffset = RefineEnvInt(
        prefix + "_LOCALITY_MIN_DEPTH_OFFSET",
        RefineEnvInt("SALT_BUNDLE_LOCALITY_MIN_DEPTH_OFFSET", std::numeric_limits<int>::min()));
    limits.maxDepthOffset = RefineEnvInt(
        prefix + "_LOCALITY_MAX_DEPTH_OFFSET",
        RefineEnvInt("SALT_BUNDLE_LOCALITY_MAX_DEPTH_OFFSET", std::numeric_limits<int>::max()));
    limits.maxBundleRootUpFromLca = RefineEnvInt(
        prefix + "_LCA_MAX_ROOT_UP",
        RefineEnvInt("SALT_BUNDLE_LCA_MAX_ROOT_UP", std::numeric_limits<int>::max()));
    limits.maxSupportParentDownFromLca = RefineEnvInt(
        prefix + "_LCA_MAX_SUPPORT_PARENT_DOWN",
        RefineEnvInt("SALT_BUNDLE_LCA_MAX_SUPPORT_PARENT_DOWN", std::numeric_limits<int>::max()));
    return limits;
}

DTYPE BundleHpwlLowerBound(const Point& rootPoint, const vector<shared_ptr<TreeNode>>& roots) {
    DTYPE minX = rootPoint.x;
    DTYPE maxX = rootPoint.x;
    DTYPE minY = rootPoint.y;
    DTYPE maxY = rootPoint.y;
    for (const auto& root : roots) {
        minX = std::min(minX, root->loc.x);
        maxX = std::max(maxX, root->loc.x);
        minY = std::min(minY, root->loc.y);
        maxY = std::max(maxY, root->loc.y);
    }
    return (maxX - minX) + (maxY - minY);
}

DTYPE BundleHpwlLowerBoundOnSupportEdge(const shared_ptr<TreeNode>& supportChild,
                                        const vector<shared_ptr<TreeNode>>& roots) {
    DTYPE minX = roots[0]->loc.x;
    DTYPE maxX = roots[0]->loc.x;
    DTYPE minY = roots[0]->loc.y;
    DTYPE maxY = roots[0]->loc.y;
    for (const auto& root : roots) {
        minX = std::min(minX, root->loc.x);
        maxX = std::max(maxX, root->loc.x);
        minY = std::min(minY, root->loc.y);
        maxY = std::max(maxY, root->loc.y);
    }

    const auto supportParent = supportChild->parent;
    if (supportParent->loc.x == supportChild->loc.x) {
        const DTYPE xSpan = AxisSpanWithPoint(minX, maxX, supportChild->loc.x);
        const DTYPE lo = std::min(supportParent->loc.y, supportChild->loc.y);
        const DTYPE hi = std::max(supportParent->loc.y, supportChild->loc.y);
        return xSpan + MinAxisSpanWithSegment(minY, maxY, lo, hi);
    }

    const DTYPE ySpan = AxisSpanWithPoint(minY, maxY, supportChild->loc.y);
    const DTYPE lo = std::min(supportParent->loc.x, supportChild->loc.x);
    const DTYPE hi = std::max(supportParent->loc.x, supportChild->loc.x);
    return MinAxisSpanWithSegment(minX, maxX, lo, hi) + ySpan;
}

DTYPE BundleBoundaryHpwlLowerBound(const vector<shared_ptr<TreeNode>>& roots) {
    if (roots.empty()) return 0;
    DTYPE minX = roots[0]->loc.x;
    DTYPE maxX = roots[0]->loc.x;
    DTYPE minY = roots[0]->loc.y;
    DTYPE maxY = roots[0]->loc.y;
    for (const auto& root : roots) {
        minX = std::min(minX, root->loc.x);
        maxX = std::max(maxX, root->loc.x);
        minY = std::min(minY, root->loc.y);
        maxY = std::max(maxY, root->loc.y);
    }
    return (maxX - minX) + (maxY - minY);
}

int BundleRootCapFor(const char* passPrefix) {
    const int globalCap = RefineEnvInt("SALT_BUNDLE_ROOT_CAP", 0);
    const std::string prefix = passPrefix ? passPrefix : "";
    return std::max(0, RefineEnvInt(prefix + "_ROOT_CAP", globalCap));
}

struct BundleRootPotential {
    int nodeId = -1;
    int slot = 0;
    DTYPE score = 0;
    int order = 0;
};

void MarkTopBundleRootPotentials(vector<unsigned char>& allowed,
                                 int slotsPerNode,
                                 vector<BundleRootPotential>& potentials,
                                 int cap) {
    if (cap <= 0 || slotsPerNode <= 0) return;
    std::sort(potentials.begin(), potentials.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.score != rhs.score) return lhs.score > rhs.score;
        return lhs.order < rhs.order;
    });
    const int keep = std::min(cap, static_cast<int>(potentials.size()));
    for (int i = 0; i < keep; ++i) {
        const auto& item = potentials[static_cast<size_t>(i)];
        if (item.nodeId < 0) continue;
        const size_t key = static_cast<size_t>(item.nodeId * slotsPerNode + item.slot);
        if (key < allowed.size()) allowed[key] = 1;
    }
}

template <size_t N>
vector<Point> SortedUniquePoints(std::array<std::pair<DTYPE, DTYPE>, N>& points, int count) {
    auto begin = points.begin();
    auto end = points.begin() + count;
    std::sort(begin, end);
    vector<Point> out;
    out.reserve(static_cast<size_t>(count));
    std::pair<DTYPE, DTYPE> last;
    bool haveLast = false;
    for (auto it = begin; it != end; ++it) {
        if (haveLast && *it == last) continue;
        out.emplace_back(it->first, it->second);
        last = *it;
        haveLast = true;
    }
    return out;
}

vector<Point> EnumerateSupportPoints(const shared_ptr<TreeNode>& supportChild,
                                     const Point& left,
                                     const Point& right) {
    auto supportParent = supportChild->parent;
    std::array<std::pair<DTYPE, DTYPE>, 4> points;
    int count = 0;
    if (supportParent->loc.x == supportChild->loc.x) {
        const DTYPE x = supportChild->loc.x;
        const DTYPE lo = std::min(supportParent->loc.y, supportChild->loc.y);
        const DTYPE hi = std::max(supportParent->loc.y, supportChild->loc.y);
        points[static_cast<size_t>(count++)] = {x, lo};
        points[static_cast<size_t>(count++)] = {x, hi};
        points[static_cast<size_t>(count++)] = {x, Clamp(left.y, lo, hi)};
        points[static_cast<size_t>(count++)] = {x, Clamp(right.y, lo, hi)};
    } else {
        const DTYPE y = supportChild->loc.y;
        const DTYPE lo = std::min(supportParent->loc.x, supportChild->loc.x);
        const DTYPE hi = std::max(supportParent->loc.x, supportChild->loc.x);
        points[static_cast<size_t>(count++)] = {lo, y};
        points[static_cast<size_t>(count++)] = {hi, y};
        points[static_cast<size_t>(count++)] = {Clamp(left.x, lo, hi), y};
        points[static_cast<size_t>(count++)] = {Clamp(right.x, lo, hi), y};
    }
    return SortedUniquePoints(points, count);
}

vector<Point> EnumerateSupportPoints(const shared_ptr<TreeNode>& supportChild,
                                     const vector<shared_ptr<TreeNode>>& roots) {
    auto supportParent = supportChild->parent;
    std::array<std::pair<DTYPE, DTYPE>, 8> points;
    int count = 0;
    if (supportParent->loc.x == supportChild->loc.x) {
        const DTYPE x = supportChild->loc.x;
        const DTYPE lo = std::min(supportParent->loc.y, supportChild->loc.y);
        const DTYPE hi = std::max(supportParent->loc.y, supportChild->loc.y);
        points[static_cast<size_t>(count++)] = {x, lo};
        points[static_cast<size_t>(count++)] = {x, hi};
        for (const auto& root : roots) points[static_cast<size_t>(count++)] = {x, Clamp(root->loc.y, lo, hi)};
    } else {
        const DTYPE y = supportChild->loc.y;
        const DTYPE lo = std::min(supportParent->loc.x, supportChild->loc.x);
        const DTYPE hi = std::max(supportParent->loc.x, supportChild->loc.x);
        points[static_cast<size_t>(count++)] = {lo, y};
        points[static_cast<size_t>(count++)] = {hi, y};
        for (const auto& root : roots) points[static_cast<size_t>(count++)] = {Clamp(root->loc.x, lo, hi), y};
    }
    return SortedUniquePoints(points, count);
}

void UpdatePathLengths(const Tree& tree, vector<DTYPE>& pathLengths) {
    tree.PreOrder([&](const shared_ptr<TreeNode>& node) {
        if (node->parent) {
            pathLengths[node->id] = pathLengths[node->parent->id] + node->WireToParent();
        } else {
            pathLengths[node->id] = 0;
        }
    });
}

void UpdateSlacks(const Tree& tree, vector<DTYPE>& pathLengths, vector<DTYPE>& slacks, double eps) {
    tree.PostOrder([&](const shared_ptr<TreeNode>& node) {
        const DTYPE ownSlack = Dist(node->loc, tree.source->loc) * (1 + eps) - pathLengths[node->id];
        if (node->children.empty()) {
            slacks[node->id] = ownSlack;
            return;
        }
        DTYPE best = ownSlack;
        for (const auto& child : node->children) best = std::min(best, slacks[child->id]);
        slacks[node->id] = best;
    });
}

void UpdateSubtreePinCounts(const Tree& tree, vector<int>& subtreePins) {
    tree.PostOrder([&](const shared_ptr<TreeNode>& node) {
        int total = node->pin ? 1 : 0;
        for (const auto& child : node->children) total += subtreePins[child->id];
        subtreePins[node->id] = total;
    });
}

void UpdateEulerTour(const shared_ptr<TreeNode>& node, vector<int>& tin, vector<int>& tout, int& timer) {
    tin[static_cast<size_t>(node->id)] = timer++;
    for (const auto& child : node->children) UpdateEulerTour(child, tin, tout, timer);
    tout[static_cast<size_t>(node->id)] = timer - 1;
}

bool IsAncestorByTour(const shared_ptr<TreeNode>& ancestor,
                      const shared_ptr<TreeNode>& descendant,
                      const vector<int>& tin,
                      const vector<int>& tout) {
    return tin[static_cast<size_t>(ancestor->id)] <= tin[static_cast<size_t>(descendant->id)] &&
           tin[static_cast<size_t>(descendant->id)] <= tout[static_cast<size_t>(ancestor->id)];
}

int LcaNodeIdByParents(shared_ptr<TreeNode> lhs,
                       shared_ptr<TreeNode> rhs,
                       const vector<int>& depths) {
    if (!lhs || !rhs) return -1;
    while (lhs && rhs && depths[static_cast<size_t>(lhs->id)] > depths[static_cast<size_t>(rhs->id)]) {
        lhs = lhs->parent;
    }
    while (lhs && rhs && depths[static_cast<size_t>(rhs->id)] > depths[static_cast<size_t>(lhs->id)]) {
        rhs = rhs->parent;
    }
    while (lhs && rhs && lhs != rhs) {
        lhs = lhs->parent;
        rhs = rhs->parent;
    }
    return (lhs && rhs && lhs == rhs) ? lhs->id : -1;
}

void ApplyTripletMove(Tree& tree, const TripletMove& move);
void ApplyQuadMove(Tree& tree, const QuadMove& move);
void ApplySiblingPairMove(Tree& tree, const PairMove& move);

long long TotalPinPathLength(Tree& tree) {
    vector<DTYPE> pathLengths(tree.UpdateId(), 0);
    UpdatePathLengths(tree, pathLengths);
    long long total = 0;
    tree.PreOrder([&](const shared_ptr<TreeNode>& node) {
        if (node->pin) total += pathLengths[node->id];
    });
    return total;
}

shared_ptr<TreeNode> SplitSupportEdge(const shared_ptr<TreeNode>& supportChild, const Point& supportPoint) {
    auto supportParent = supportChild->parent;
    if (supportPoint == supportParent->loc) return supportParent;
    if (supportPoint == supportChild->loc) return supportChild;

    TreeNode::ResetParent(supportChild);
    auto split = make_shared<TreeNode>(supportPoint);
    TreeNode::SetParent(split, supportParent);
    TreeNode::SetParent(supportChild, split);
    return split;
}

void MaterializeThreePointConnector(const shared_ptr<TreeNode>& attach,
                                    const shared_ptr<TreeNode>& left,
                                    const shared_ptr<TreeNode>& right,
                                    const Point& branchPoint) {
    if (branchPoint == attach->loc) {
        TreeNode::SetParent(left, attach);
        TreeNode::SetParent(right, attach);
        return;
    }
    if (branchPoint == left->loc) {
        TreeNode::SetParent(left, attach);
        TreeNode::SetParent(right, left);
        return;
    }
    if (branchPoint == right->loc) {
        TreeNode::SetParent(right, attach);
        TreeNode::SetParent(left, right);
        return;
    }

    auto branch = make_shared<TreeNode>(branchPoint);
    TreeNode::SetParent(branch, attach);
    TreeNode::SetParent(left, branch);
    TreeNode::SetParent(right, branch);
}

bool HasDuplicateBundlePoints(const Point& rootPoint, const vector<shared_ptr<TreeNode>>& roots) {
    for (size_t i = 0; i < roots.size(); ++i) {
        if (roots[i]->loc == rootPoint) return true;
        for (size_t j = 0; j < i; ++j) {
            if (roots[i]->loc == roots[j]->loc) return true;
        }
    }
    return false;
}

LocalFluteSolution BuildLocalTreeFromEdges(const vector<Point>& inputPoints,
                                           const vector<int>& inputPinIds,
                                           int terminalCount,
                                           const vector<std::pair<int, int>>& rawEdges) {
    LocalFluteSolution out;
    if (terminalCount < 2 || rawEdges.empty() || inputPoints.size() != inputPinIds.size()) return out;

    vector<Point> nodePoints = inputPoints;
    vector<int> nodePinIds = inputPinIds;
    vector<Point> terminalPoints(static_cast<size_t>(terminalCount));
    vector<unsigned char> terminalSeen(static_cast<size_t>(terminalCount), 0);
    int sourceVertex = -1;
    for (int idx = 0; idx < static_cast<int>(nodePinIds.size()); ++idx) {
        const int pinId = nodePinIds[static_cast<size_t>(idx)];
        if (pinId < 0) continue;
        if (pinId >= terminalCount || terminalSeen[static_cast<size_t>(pinId)]) return out;
        terminalPoints[static_cast<size_t>(pinId)] = nodePoints[static_cast<size_t>(idx)];
        terminalSeen[static_cast<size_t>(pinId)] = 1;
        if (pinId == 0) sourceVertex = idx;
    }
    if (sourceVertex < 0) return out;
    for (unsigned char seenTerminal : terminalSeen) {
        if (!seenTerminal) return out;
    }

    vector<std::set<int>> adj;
    adj.resize(nodePoints.size());
    auto ensure_adj_size = [&]() {
        if (adj.size() < nodePoints.size()) adj.resize(nodePoints.size());
    };
    for (const auto& edge : rawEdges) {
        const int u = edge.first;
        const int v = edge.second;
        if (u < 0 || v < 0 || u >= static_cast<int>(nodePoints.size()) ||
            v >= static_cast<int>(nodePoints.size()) || u == v) {
            continue;
        }
        ensure_adj_size();
        adj[static_cast<size_t>(u)].insert(v);
        adj[static_cast<size_t>(v)].insert(u);
    }

    vector<unsigned char> seen(static_cast<size_t>(nodePoints.size()), 0);
    int seenCount = 0;
    function<void(int)> mark = [&](int idx) {
        if (seen[static_cast<size_t>(idx)]) return;
        seen[static_cast<size_t>(idx)] = 1;
        ++seenCount;
        for (int next : adj[static_cast<size_t>(idx)]) mark(next);
    };
    mark(sourceVertex);
    for (int tid = 0; tid < terminalCount; ++tid) {
        bool terminalReachable = false;
        for (int idx = 0; idx < static_cast<int>(nodePinIds.size()); ++idx) {
            if (nodePinIds[static_cast<size_t>(idx)] == tid && seen[static_cast<size_t>(idx)]) {
                terminalReachable = true;
                break;
            }
        }
        if (!terminalReachable) return out;
    }
    int edgeCount = 0;
    for (int idx = 0; idx < static_cast<int>(adj.size()); ++idx) {
        if (!seen[static_cast<size_t>(idx)]) continue;
        edgeCount += static_cast<int>(adj[static_cast<size_t>(idx)].size());
    }
    edgeCount /= 2;
    if (edgeCount != seenCount - 1) return out;

    vector<shared_ptr<Pin>> localPins(static_cast<size_t>(terminalCount));
    for (int tid = 0; tid < terminalCount; ++tid) {
        localPins[static_cast<size_t>(tid)] =
            make_shared<Pin>(terminalPoints[static_cast<size_t>(tid)], tid);
    }

    function<shared_ptr<TreeNode>(int, int)> build = [&](int idx, int parentIdx) {
        const int pinId = nodePinIds[static_cast<size_t>(idx)];
        auto node = pinId >= 0 ? make_shared<TreeNode>(localPins[static_cast<size_t>(pinId)])
                               : make_shared<TreeNode>(nodePoints[static_cast<size_t>(idx)]);
        for (int next : adj[static_cast<size_t>(idx)]) {
            if (next == parentIdx) continue;
            auto child = build(next, idx);
            TreeNode::SetParent(child, node);
        }
        return node;
    };

    out.tree.source = build(sourceVertex, -1);
    out.tree.RemovePhyRedundantSteiner();
    out.tree.RemoveTopoRedundantSteiner();
    out.tree.RemoveEmptyChildren();
    out.wl = WireLengthEvalBase(out.tree).wireLength;

    out.sinkPathLengths.assign(static_cast<size_t>(terminalCount - 1), 0);
    vector<unsigned char> seenSink(static_cast<size_t>(terminalCount - 1), 0);
    function<void(const shared_ptr<TreeNode>&, DTYPE)> dfs =
        [&](const shared_ptr<TreeNode>& node, DTYPE dist) {
            if (node->pin && node->pin->id > 0) {
                const size_t idx = static_cast<size_t>(node->pin->id - 1);
                if (idx < seenSink.size()) {
                    out.sinkPathLengths[idx] = dist;
                    seenSink[idx] = 1;
                }
            }
            for (const auto& child : node->children) dfs(child, dist + child->WireToParent());
        };
    dfs(out.tree.source, 0);
    for (unsigned char flag : seenSink) {
        if (!flag) {
            out.tree.Reset();
            out.sinkPathLengths.clear();
            return out;
        }
    }
    out.valid = true;
    return out;
}

LocalFluteSolution SolveLocalFourPointEnum(const Point& rootPoint, const vector<shared_ptr<TreeNode>>& roots) {
    LocalFluteSolution best;
    if (roots.size() != 3 || HasDuplicateBundlePoints(rootPoint, roots)) return best;

    vector<Point> terminals = {rootPoint, roots[0]->loc, roots[1]->loc, roots[2]->loc};
    const std::array<std::array<int, 4>, 3> pairings = {{
        {{0, 1, 2, 3}},
        {{0, 2, 1, 3}},
        {{0, 3, 1, 2}},
    }};
    auto find_or_add_point = [](vector<Point>& points, vector<int>& pinIds, const Point& point) {
        for (int idx = 0; idx < static_cast<int>(points.size()); ++idx) {
            if (points[static_cast<size_t>(idx)] == point) return idx;
        }
        points.push_back(point);
        pinIds.push_back(-1);
        return static_cast<int>(points.size()) - 1;
    };
    struct FourPointCandidate {
        bool valid = false;
        DTYPE wl = std::numeric_limits<DTYPE>::max();
        long long pathSum = std::numeric_limits<long long>::max();
        vector<DTYPE> sinkPaths;
        vector<Point> points;
        vector<int> pinIds;
        vector<std::pair<int, int>> edges;
    };
    FourPointCandidate bestCandidate;
    auto evaluate_candidate = [&](vector<Point> candidatePoints,
                                  vector<int> pinIds,
                                  vector<std::pair<int, int>> rawEdges) {
        FourPointCandidate cand;
        const int n = static_cast<int>(candidatePoints.size());
        vector<vector<int>> adj(static_cast<size_t>(n));
        vector<std::pair<int, int>> edges;
        auto add_edge = [&](int u, int v) {
            if (u < 0 || v < 0 || u >= n || v >= n || u == v) return;
            if (u > v) std::swap(u, v);
            for (const auto& edge : edges) {
                if (edge.first == u && edge.second == v) return;
            }
            edges.push_back({u, v});
            adj[static_cast<size_t>(u)].push_back(v);
            adj[static_cast<size_t>(v)].push_back(u);
        };
        for (auto edge : rawEdges) add_edge(edge.first, edge.second);

        int sourceVertex = -1;
        vector<int> terminalVertex(4, -1);
        for (int idx = 0; idx < n; ++idx) {
            const int pinId = pinIds[static_cast<size_t>(idx)];
            if (pinId < 0) continue;
            if (pinId >= 4 || terminalVertex[static_cast<size_t>(pinId)] >= 0) return cand;
            terminalVertex[static_cast<size_t>(pinId)] = idx;
            if (pinId == 0) sourceVertex = idx;
        }
        if (sourceVertex < 0) return cand;
        for (int idx = 0; idx < 4; ++idx) {
            if (terminalVertex[static_cast<size_t>(idx)] < 0) return cand;
        }

        vector<unsigned char> seen(static_cast<size_t>(n), 0);
        vector<DTYPE> dist(static_cast<size_t>(n), 0);
        int seenCount = 0;
        function<void(int)> dfs = [&](int u) {
            seen[static_cast<size_t>(u)] = 1;
            ++seenCount;
            for (int v : adj[static_cast<size_t>(u)]) {
                if (seen[static_cast<size_t>(v)]) continue;
                dist[static_cast<size_t>(v)] =
                    dist[static_cast<size_t>(u)] + Dist(candidatePoints[static_cast<size_t>(u)],
                                                        candidatePoints[static_cast<size_t>(v)]);
                dfs(v);
            }
        };
        dfs(sourceVertex);
        for (int pinId = 0; pinId < 4; ++pinId) {
            if (!seen[static_cast<size_t>(terminalVertex[static_cast<size_t>(pinId)])]) return cand;
        }
        int reachableEdges = 0;
        cand.wl = 0;
        for (const auto& edge : edges) {
            if (!seen[static_cast<size_t>(edge.first)] || !seen[static_cast<size_t>(edge.second)]) continue;
            ++reachableEdges;
            cand.wl += Dist(candidatePoints[static_cast<size_t>(edge.first)],
                            candidatePoints[static_cast<size_t>(edge.second)]);
        }
        if (reachableEdges != seenCount - 1) return cand;

        cand.sinkPaths.assign(3, 0);
        cand.pathSum = 0;
        for (int pinId = 1; pinId < 4; ++pinId) {
            const DTYPE path = dist[static_cast<size_t>(terminalVertex[static_cast<size_t>(pinId)])];
            cand.sinkPaths[static_cast<size_t>(pinId - 1)] = path;
            cand.pathSum += path;
        }
        cand.valid = true;
        cand.points = std::move(candidatePoints);
        cand.pinIds = std::move(pinIds);
        cand.edges = std::move(edges);
        return cand;
    };
    auto append_interval_endpoints = [](vector<DTYPE>& values, DTYPE lo, DTYPE hi) {
        if (lo > hi) std::swap(lo, hi);
        values.push_back(lo);
        values.push_back(hi);
    };

    for (const auto& pairing : pairings) {
        const Point& a = terminals[static_cast<size_t>(pairing[0])];
        const Point& b = terminals[static_cast<size_t>(pairing[1])];
        const Point& c = terminals[static_cast<size_t>(pairing[2])];
        const Point& d = terminals[static_cast<size_t>(pairing[3])];
        const DTYPE xGap = std::max<DTYPE>(0, std::min(std::max(a.x, b.x), std::max(c.x, d.x)) -
                                               std::max(std::min(a.x, b.x), std::min(c.x, d.x)));
        const DTYPE yGap = std::max<DTYPE>(0, std::min(std::max(a.y, b.y), std::max(c.y, d.y)) -
                                               std::max(std::min(a.y, b.y), std::min(c.y, d.y)));
        vector<Point> sCandidates;
        vector<Point> tCandidates;
        if (xGap <= yGap) {
            vector<DTYPE> ys;
            append_interval_endpoints(ys,
                                      std::max(std::min(a.y, b.y), std::min(c.y, d.y)),
                                      std::min(std::max(a.y, b.y), std::max(c.y, d.y)));
            ys.push_back(a.y);
            ys.push_back(b.y);
            ys.push_back(c.y);
            ys.push_back(d.y);
            std::sort(ys.begin(), ys.end());
            ys.erase(std::unique(ys.begin(), ys.end()), ys.end());
            const DTYPE sx = Median3(a.x, b.x, c.x);
            const DTYPE tx = Median3(c.x, d.x, a.x);
            for (DTYPE y : ys) {
                sCandidates.emplace_back(sx, Median3(a.y, b.y, y));
                tCandidates.emplace_back(tx, Median3(c.y, d.y, y));
            }
        } else {
            vector<DTYPE> xs;
            append_interval_endpoints(xs,
                                      std::max(std::min(a.x, b.x), std::min(c.x, d.x)),
                                      std::min(std::max(a.x, b.x), std::max(c.x, d.x)));
            xs.push_back(a.x);
            xs.push_back(b.x);
            xs.push_back(c.x);
            xs.push_back(d.x);
            std::sort(xs.begin(), xs.end());
            xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
            const DTYPE sy = Median3(a.y, b.y, c.y);
            const DTYPE ty = Median3(c.y, d.y, a.y);
            for (DTYPE x : xs) {
                sCandidates.emplace_back(Median3(a.x, b.x, x), sy);
                tCandidates.emplace_back(Median3(c.x, d.x, x), ty);
            }
        }
        for (size_t idx = 0; idx < sCandidates.size(); ++idx) {
            vector<Point> candidatePoints = terminals;
            vector<int> pinIds = {0, 1, 2, 3};
            const int sIdx = find_or_add_point(candidatePoints, pinIds, sCandidates[idx]);
            const int tIdx = find_or_add_point(candidatePoints, pinIds, tCandidates[idx]);
            vector<std::pair<int, int>> edges = {
                {pairing[0], sIdx},
                {pairing[1], sIdx},
                {sIdx, tIdx},
                {pairing[2], tIdx},
                {pairing[3], tIdx},
            };
            auto cand = evaluate_candidate(std::move(candidatePoints), std::move(pinIds), std::move(edges));
            if (!cand.valid) continue;
            if (!bestCandidate.valid || cand.wl < bestCandidate.wl ||
                (cand.wl == bestCandidate.wl && cand.pathSum < bestCandidate.pathSum)) {
                bestCandidate = std::move(cand);
            }
        }
    }
    if (!bestCandidate.valid) return best;
    LocalFluteSolution built =
        BuildLocalTreeFromEdges(bestCandidate.points, bestCandidate.pinIds, 4, bestCandidate.edges);
    if (!built.valid) return best;
    best.valid = built.valid;
    best.wl = built.wl;
    best.sinkPathLengths = std::move(built.sinkPathLengths);
    best.tree.source = built.tree.source;
    best.tree.net = built.tree.net;
    built.tree.source = nullptr;
    built.tree.net = nullptr;
    return best;
}

LocalFluteSolution SolveLocalFourPointFast(const Point& rootPoint, const vector<shared_ptr<TreeNode>>& roots) {
    return SolveLocalFourPointEnum(rootPoint, roots);
}

struct HananParent {
    enum Kind { kNone, kTerminal, kMerge, kEdge } kind = kNone;
    int a = -1;
};

LocalFluteSolution SolveLocalHananDp(const Point& rootPoint, const vector<shared_ptr<TreeNode>>& roots) {
    LocalFluteSolution out;
    if (roots.size() < 2 || HasDuplicateBundlePoints(rootPoint, roots)) return out;

    vector<Point> terminals;
    terminals.reserve(roots.size() + 1);
    terminals.push_back(rootPoint);
    for (const auto& root : roots) terminals.push_back(root->loc);
    const int terminalCount = static_cast<int>(terminals.size());
    if (terminalCount <= 1 || terminalCount > 8) return out;

    vector<DTYPE> xs;
    vector<DTYPE> ys;
    xs.reserve(terminals.size());
    ys.reserve(terminals.size());
    for (const auto& point : terminals) {
        xs.push_back(point.x);
        ys.push_back(point.y);
    }
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    std::sort(ys.begin(), ys.end());
    ys.erase(std::unique(ys.begin(), ys.end()), ys.end());
    if (xs.empty() || ys.empty()) return out;

    const int nx = static_cast<int>(xs.size());
    const int ny = static_cast<int>(ys.size());
    const int vertexCount = nx * ny;
    auto vertex_id = [&](int xi, int yi) { return yi * nx + xi; };
    auto point_for_vertex = [&](int id) {
        const int xi = id % nx;
        const int yi = id / nx;
        return Point(xs[static_cast<size_t>(xi)], ys[static_cast<size_t>(yi)]);
    };
    auto coord_index = [](const vector<DTYPE>& values, DTYPE value) {
        auto it = std::lower_bound(values.begin(), values.end(), value);
        return static_cast<int>(it - values.begin());
    };

    vector<vector<pair<int, DTYPE>>> graph(static_cast<size_t>(vertexCount));
    auto add_edge = [&](int u, int v, DTYPE w) {
        graph[static_cast<size_t>(u)].push_back({v, w});
        graph[static_cast<size_t>(v)].push_back({u, w});
    };
    for (int yi = 0; yi < ny; ++yi) {
        for (int xi = 1; xi < nx; ++xi) {
            add_edge(vertex_id(xi - 1, yi), vertex_id(xi, yi),
                     xs[static_cast<size_t>(xi)] - xs[static_cast<size_t>(xi - 1)]);
        }
    }
    for (int xi = 0; xi < nx; ++xi) {
        for (int yi = 1; yi < ny; ++yi) {
            add_edge(vertex_id(xi, yi - 1), vertex_id(xi, yi),
                     ys[static_cast<size_t>(yi)] - ys[static_cast<size_t>(yi - 1)]);
        }
    }

    vector<int> terminalVertex(static_cast<size_t>(terminalCount), -1);
    vector<int> terminalAtVertex(static_cast<size_t>(vertexCount), -1);
    for (int tid = 0; tid < terminalCount; ++tid) {
        const int xi = coord_index(xs, terminals[static_cast<size_t>(tid)].x);
        const int yi = coord_index(ys, terminals[static_cast<size_t>(tid)].y);
        if (xi < 0 || xi >= nx || yi < 0 || yi >= ny) return out;
        const int vid = vertex_id(xi, yi);
        terminalVertex[static_cast<size_t>(tid)] = vid;
        terminalAtVertex[static_cast<size_t>(vid)] = tid;
    }

    const int fullMask = (1 << terminalCount) - 1;
    const int kInf = std::numeric_limits<int>::max() / 4;
    vector<vector<DTYPE>> dp(static_cast<size_t>(fullMask + 1),
                             vector<DTYPE>(static_cast<size_t>(vertexCount), kInf));
    vector<vector<HananParent>> parent(static_cast<size_t>(fullMask + 1),
                                       vector<HananParent>(static_cast<size_t>(vertexCount)));

    for (int tid = 0; tid < terminalCount; ++tid) {
        const int mask = 1 << tid;
        const int vid = terminalVertex[static_cast<size_t>(tid)];
        dp[static_cast<size_t>(mask)][static_cast<size_t>(vid)] = 0;
        parent[static_cast<size_t>(mask)][static_cast<size_t>(vid)].kind = HananParent::kTerminal;
    }

    for (int mask = 1; mask <= fullMask; ++mask) {
        for (int sub = (mask - 1) & mask; sub > 0; sub = (sub - 1) & mask) {
            const int other = mask ^ sub;
            if (other == 0) continue;
            for (int v = 0; v < vertexCount; ++v) {
                const DTYPE left = dp[static_cast<size_t>(sub)][static_cast<size_t>(v)];
                const DTYPE right = dp[static_cast<size_t>(other)][static_cast<size_t>(v)];
                if (left >= kInf || right >= kInf) continue;
                const DTYPE merged = left + right;
                if (merged < dp[static_cast<size_t>(mask)][static_cast<size_t>(v)]) {
                    dp[static_cast<size_t>(mask)][static_cast<size_t>(v)] = merged;
                    parent[static_cast<size_t>(mask)][static_cast<size_t>(v)].kind = HananParent::kMerge;
                    parent[static_cast<size_t>(mask)][static_cast<size_t>(v)].a = sub;
                }
            }
        }

        vector<unsigned char> used(static_cast<size_t>(vertexCount), 0);
        for (int iter = 0; iter < vertexCount; ++iter) {
            int v = -1;
            DTYPE bestDist = kInf;
            for (int cand = 0; cand < vertexCount; ++cand) {
                if (used[static_cast<size_t>(cand)]) continue;
                const DTYPE dist = dp[static_cast<size_t>(mask)][static_cast<size_t>(cand)];
                if (dist < bestDist) {
                    bestDist = dist;
                    v = cand;
                }
            }
            if (v < 0 || bestDist >= kInf) break;
            used[static_cast<size_t>(v)] = 1;
            for (const auto& edge : graph[static_cast<size_t>(v)]) {
                const int to = edge.first;
                const DTYPE relaxed = bestDist + edge.second;
                if (relaxed < dp[static_cast<size_t>(mask)][static_cast<size_t>(to)]) {
                    dp[static_cast<size_t>(mask)][static_cast<size_t>(to)] = relaxed;
                    parent[static_cast<size_t>(mask)][static_cast<size_t>(to)].kind = HananParent::kEdge;
                    parent[static_cast<size_t>(mask)][static_cast<size_t>(to)].a = v;
                }
            }
        }
    }

    const int sourceVertex = terminalVertex[0];
    if (dp[static_cast<size_t>(fullMask)][static_cast<size_t>(sourceVertex)] >= kInf) return out;

    std::set<std::pair<int, int>> edgeSet;
    vector<unsigned char> visiting(static_cast<size_t>((fullMask + 1) * vertexCount), 0);
    function<bool(int, int)> reconstruct = [&](int mask, int vertex) -> bool {
        const size_t state = static_cast<size_t>(mask * vertexCount + vertex);
        if (visiting[state]) return true;
        visiting[state] = 1;
        const HananParent& par = parent[static_cast<size_t>(mask)][static_cast<size_t>(vertex)];
        if (par.kind == HananParent::kTerminal) return true;
        if (par.kind == HananParent::kEdge) {
            if (par.a < 0) return false;
            if (!reconstruct(mask, par.a)) return false;
            edgeSet.insert(std::minmax(par.a, vertex));
            return true;
        }
        if (par.kind == HananParent::kMerge) {
            const int sub = par.a;
            if (sub <= 0 || (sub & mask) != sub) return false;
            const int other = mask ^ sub;
            if (other <= 0) return false;
            return reconstruct(sub, vertex) && reconstruct(other, vertex);
        }
        return false;
    };
    if (!reconstruct(fullMask, sourceVertex) || edgeSet.empty()) return out;

    vector<vector<int>> treeAdj(static_cast<size_t>(vertexCount));
    for (const auto& edge : edgeSet) {
        treeAdj[static_cast<size_t>(edge.first)].push_back(edge.second);
        treeAdj[static_cast<size_t>(edge.second)].push_back(edge.first);
    }

    vector<shared_ptr<Pin>> localPins(static_cast<size_t>(terminalCount));
    for (int tid = 0; tid < terminalCount; ++tid) {
        localPins[static_cast<size_t>(tid)] =
            make_shared<Pin>(terminals[static_cast<size_t>(tid)], tid);
    }

    function<shared_ptr<TreeNode>(int, int)> build_node = [&](int vertex, int parentVertex) {
        const int tid = terminalAtVertex[static_cast<size_t>(vertex)];
        auto node = make_shared<TreeNode>(point_for_vertex(vertex),
                                          tid >= 0 ? localPins[static_cast<size_t>(tid)] : nullptr);
        for (int childVertex : treeAdj[static_cast<size_t>(vertex)]) {
            if (childVertex == parentVertex) continue;
            auto child = build_node(childVertex, vertex);
            TreeNode::SetParent(child, node);
        }
        return node;
    };

    out.tree.source = build_node(sourceVertex, -1);
    out.tree.RemovePhyRedundantSteiner();
    out.tree.RemoveTopoRedundantSteiner();
    out.tree.RemoveEmptyChildren();
    out.wl = WireLengthEvalBase(out.tree).wireLength;

    out.sinkPathLengths.assign(roots.size(), 0);
    vector<unsigned char> seenSink(roots.size(), 0);
    function<void(const shared_ptr<TreeNode>&, DTYPE)> dfs =
        [&](const shared_ptr<TreeNode>& node, DTYPE dist) {
            if (node->pin && node->pin->id > 0) {
                const size_t idx = static_cast<size_t>(node->pin->id - 1);
                if (idx < seenSink.size()) {
                    out.sinkPathLengths[idx] = dist;
                    seenSink[idx] = 1;
                }
            }
            for (const auto& child : node->children) dfs(child, dist + child->WireToParent());
        };
    dfs(out.tree.source, 0);
    for (unsigned char seen : seenSink) {
        if (!seen) {
            out.tree.Reset();
            out.sinkPathLengths.clear();
            return out;
        }
    }
    out.valid = true;
    return out;
}

bool UseLocalHananDp() {
    if (RefineEnvInt("SALT_LOCAL_BUNDLE_HANAN_DP", 0) != 0) return true;
    const char* solver = std::getenv("SALT_LOCAL_BUNDLE_SOLVER");
    return solver && std::string(solver) == "hanan_dp";
}

bool UseLocalHananDpForTerminalCount(int terminalCount) {
    if (!UseLocalHananDp()) return false;
    const int minTerminals = std::max(2, RefineEnvInt("SALT_LOCAL_BUNDLE_HANAN_DP_MIN_TERMINALS", 2));
    const int maxTerminals = std::max(0, RefineEnvInt("SALT_LOCAL_BUNDLE_HANAN_DP_MAX_TERMINALS", 8));
    return terminalCount >= minTerminals && terminalCount <= maxTerminals;
}

LocalFluteSolution SolveLocalFlute(const Point& rootPoint, const vector<shared_ptr<TreeNode>>& roots) {
    LocalFluteSolution out;
    if (roots.size() < 2 || HasDuplicateBundlePoints(rootPoint, roots)) return out;

    if (roots.size() == 3 && RefineEnvInt("SALT_LOCAL_BUNDLE_FAST4", 0) != 0) {
        LocalFluteSolution fast4 = SolveLocalFourPointEnum(rootPoint, roots);
        if (fast4.valid || RefineEnvInt("SALT_LOCAL_BUNDLE_FAST4_STRICT", 0) != 0) {
            out.valid = fast4.valid;
            out.wl = fast4.wl;
            out.sinkPathLengths = std::move(fast4.sinkPathLengths);
            out.tree.source = fast4.tree.source;
            out.tree.net = fast4.tree.net;
            fast4.tree.source = nullptr;
            fast4.tree.net = nullptr;
            return out;
        }
    }

    if (UseLocalHananDpForTerminalCount(static_cast<int>(roots.size()) + 1)) {
        LocalFluteSolution hanan = SolveLocalHananDp(rootPoint, roots);
        if (hanan.valid || RefineEnvInt("SALT_LOCAL_BUNDLE_HANAN_DP_STRICT", 0) != 0) {
            out.valid = hanan.valid;
            out.wl = hanan.wl;
            out.sinkPathLengths = std::move(hanan.sinkPathLengths);
            out.tree.source = hanan.tree.source;
            out.tree.net = hanan.tree.net;
            hanan.tree.source = nullptr;
            hanan.tree.net = nullptr;
            return out;
        }
    }

    salt::Net localNet;
    localNet.id = -1;
    localNet.name = "local_bundle";
    localNet.pins.reserve(roots.size() + 1);
    localNet.pins.push_back(make_shared<Pin>(rootPoint, 0));
    for (size_t idx = 0; idx < roots.size(); ++idx) {
        localNet.pins.push_back(make_shared<Pin>(roots[idx]->loc, static_cast<int>(idx + 1)));
    }

    FluteBuilder fluteBuilder;
    fluteBuilder.Run(localNet, out.tree);
    out.tree.RemovePhyRedundantSteiner();
    out.tree.RemoveTopoRedundantSteiner();
    out.tree.RemoveEmptyChildren();
    out.wl = WireLengthEvalBase(out.tree).wireLength;

    out.sinkPathLengths.assign(roots.size(), 0);
    function<void(const shared_ptr<TreeNode>&, DTYPE)> dfs = [&](const shared_ptr<TreeNode>& node, DTYPE dist) {
        if (node->pin && node->pin->id > 0) {
            out.sinkPathLengths[static_cast<size_t>(node->pin->id - 1)] = dist;
        }
        for (const auto& child : node->children) dfs(child, dist + child->WireToParent());
    };
    dfs(out.tree.source, 0);
    out.valid = true;
    return out;
}

shared_ptr<TreeNode> CloneLocalBundleNode(const shared_ptr<TreeNode>& node,
                                          const vector<shared_ptr<TreeNode>>& roots) {
    shared_ptr<TreeNode> clone;
    if (node->pin && node->pin->id > 0) {
        clone = roots[static_cast<size_t>(node->pin->id - 1)];
    } else {
        clone = make_shared<TreeNode>(node->loc);
    }
    for (const auto& child : node->children) {
        auto childClone = CloneLocalBundleNode(child, roots);
        TreeNode::SetParent(childClone, clone);
    }
    return clone;
}

void MaterializeLocalFluteAtAttach(const shared_ptr<TreeNode>& attach,
                                   const shared_ptr<TreeNode>& localSource,
                                   const vector<shared_ptr<TreeNode>>& roots) {
    for (const auto& child : localSource->children) {
        auto childClone = CloneLocalBundleNode(child, roots);
        TreeNode::SetParent(childClone, attach);
    }
}

bool BetterLexicographic(DTYPE delta,
                         long long weightedPathDelta,
                         DTYPE bestDelta,
                         long long bestWeightedPathDelta) {
    return (delta < bestDelta) || (delta == bestDelta && weightedPathDelta < bestWeightedPathDelta);
}

void UpdateDepths(const Tree& tree, vector<int>& depths) {
    tree.PreOrder([&](const shared_ptr<TreeNode>& node) {
        depths[node->id] = node->parent ? depths[node->parent->id] + 1 : 0;
    });
}

bool BundleSupportEdgeAllowed(const shared_ptr<TreeNode>& bundleRoot,
                              const shared_ptr<TreeNode>& supportChild,
                              const vector<DTYPE>& pathLengths,
                              const vector<int>& depths,
                              const BundleLocalityLimits& limits,
                              const vector<unsigned char>* lcaAllowed = nullptr,
                              int lcaAllowedStride = 0) {
    if (!limits.enabled) return true;
    if (!bundleRoot || !supportChild) return false;
    const int depthOffset = depths[supportChild->id] - depths[bundleRoot->id];
    if (depthOffset < limits.minDepthOffset || depthOffset > limits.maxDepthOffset) return false;
    if (limits.maxPathWlSpan >= 0 &&
        std::abs(pathLengths[supportChild->id] - pathLengths[bundleRoot->id]) > limits.maxPathWlSpan) {
        return false;
    }
    if (limits.lcaEnabled) {
        if (lcaAllowed && !lcaAllowed->empty() && lcaAllowedStride > 0) {
            const size_t key =
                static_cast<size_t>(bundleRoot->id) * static_cast<size_t>(lcaAllowedStride) +
                static_cast<size_t>(supportChild->id);
            return key < lcaAllowed->size() && (*lcaAllowed)[key] != 0;
        }
        const auto supportParent = supportChild->parent;
        if (!supportParent) return false;
        const int lca = LcaNodeIdByParents(bundleRoot, supportParent, depths);
        if (lca < 0) return false;
        const int rootUp = depths[static_cast<size_t>(bundleRoot->id)] - depths[static_cast<size_t>(lca)];
        const int supportParentDown =
            depths[static_cast<size_t>(supportParent->id)] - depths[static_cast<size_t>(lca)];
        if (rootUp > limits.maxBundleRootUpFromLca) return false;
        if (supportParentDown > limits.maxSupportParentDownFromLca) return false;
    }
    return true;
}

vector<unsigned char> BuildBundleLcaLocalityMatrix(const vector<shared_ptr<TreeNode>>& nodes,
                                                   int numNodes,
                                                   const vector<int>& depths,
                                                   const BundleLocalityLimits& limits) {
    vector<unsigned char> allowed;
    if (!limits.lcaEnabled || numNodes <= 0) return allowed;
    allowed.assign(static_cast<size_t>(numNodes) * static_cast<size_t>(numNodes), 0);
    for (const auto& bundleRoot : nodes) {
        if (!bundleRoot || bundleRoot->id < 0 || bundleRoot->id >= numNodes) continue;
        for (const auto& supportChild : nodes) {
            if (!supportChild || !supportChild->parent ||
                supportChild->id < 0 || supportChild->id >= numNodes) {
                continue;
            }
            const int lca = LcaNodeIdByParents(bundleRoot, supportChild->parent, depths);
            if (lca < 0) continue;
            const int rootUp = depths[static_cast<size_t>(bundleRoot->id)] - depths[static_cast<size_t>(lca)];
            const int supportParentDown =
                depths[static_cast<size_t>(supportChild->parent->id)] - depths[static_cast<size_t>(lca)];
            if (rootUp > limits.maxBundleRootUpFromLca ||
                supportParentDown > limits.maxSupportParentDownFromLca) {
                continue;
            }
            allowed[static_cast<size_t>(bundleRoot->id) * static_cast<size_t>(numNodes) +
                    static_cast<size_t>(supportChild->id)] = 1;
        }
    }
    return allowed;
}

vector<vector<shared_ptr<TreeNode>>> BuildBundleLcaSupportLists(
    const vector<shared_ptr<TreeNode>>& roots,
    const vector<shared_ptr<TreeNode>>& supports,
    int numNodes,
    const vector<int>& depths,
    const BundleLocalityLimits& limits) {
    vector<vector<shared_ptr<TreeNode>>> lists;
    if (!limits.lcaEnumerate || numNodes <= 0) return lists;
    lists.resize(static_cast<size_t>(numNodes));
    const int maxRootUp = std::max(0, limits.maxBundleRootUpFromLca);
    const int maxSupportDown = std::max(0, limits.maxSupportParentDownFromLca);

    vector<vector<shared_ptr<TreeNode>>> childrenByParent(static_cast<size_t>(numNodes));
    for (const auto& node : supports) {
        if (!node || !node->parent || node->parent->id < 0 || node->parent->id >= numNodes) continue;
        childrenByParent[static_cast<size_t>(node->parent->id)].push_back(node);
    }

    vector<int> visited(static_cast<size_t>(numNodes), -1);
    vector<shared_ptr<TreeNode>> stack;
    int stamp = 0;
    for (const auto& root : roots) {
        if (!root || root->id < 0 || root->id >= numNodes) continue;
        auto& out = lists[static_cast<size_t>(root->id)];
        ++stamp;
        shared_ptr<TreeNode> lca = root;
        for (int up = 0; lca && up <= maxRootUp; ++up, lca = lca->parent) {
            stack.clear();
            stack.push_back(lca);
            while (!stack.empty()) {
                auto parent = stack.back();
                stack.pop_back();
                if (!parent || parent->id < 0 || parent->id >= numNodes) continue;
                const int supportDown =
                    depths[static_cast<size_t>(parent->id)] - depths[static_cast<size_t>(lca->id)];
                if (supportDown > maxSupportDown) continue;
                for (const auto& child : childrenByParent[static_cast<size_t>(parent->id)]) {
                    if (!child || child->id < 0 || child->id >= numNodes) continue;
                    if (visited[static_cast<size_t>(child->id)] != stamp) {
                        visited[static_cast<size_t>(child->id)] = stamp;
                        out.push_back(child);
                    }
                    if (supportDown < maxSupportDown) stack.push_back(child);
                }
            }
        }
    }
    return lists;
}

const vector<shared_ptr<TreeNode>>& BundleSupportCandidatesFor(
    const shared_ptr<TreeNode>& node,
    const vector<shared_ptr<TreeNode>>& fallback,
    const vector<vector<shared_ptr<TreeNode>>>& lcaSupportLists) {
    if (!node || lcaSupportLists.empty() ||
        node->id < 0 || node->id >= static_cast<int>(lcaSupportLists.size())) {
        return fallback;
    }
    return lcaSupportLists[static_cast<size_t>(node->id)];
}

TripletMove FindBestTripletMove(Tree& tree, double eps, bool allowNeutral) {
    const int numNodes = tree.UpdateId();
    auto nodes = tree.ObtainNodes();
    vector<DTYPE> pathLengths(numNodes, 0);
    vector<DTYPE> slacks(numNodes, 0);
    vector<int> subtreePins(numNodes, 0);
    UpdatePathLengths(tree, pathLengths);
    UpdateSlacks(tree, pathLengths, slacks, eps);
    UpdateSubtreePinCounts(tree, subtreePins);
    vector<int> tin(static_cast<size_t>(numNodes), 0);
    vector<int> tout(static_cast<size_t>(numNodes), 0);
    int tourTimer = 0;
    UpdateEulerTour(tree.source, tin, tout, tourTimer);

    TripletMove best;
    const int fastMargin = std::max(0, RefineEnvInt("SALT_TRIPLET_FAST4_MARGIN", 0));
    for (const auto& node : nodes) {
        if (!node->parent || node->pin || node->children.size() != 2) continue;
        for (int innerIdx = 0; innerIdx < 2; ++innerIdx) {
            auto inner = node->children[static_cast<size_t>(innerIdx)];
            auto side = node->children[static_cast<size_t>(1 - innerIdx)];
            if (inner->pin || inner->children.size() != 2) continue;

            vector<shared_ptr<TreeNode>> roots = {side, inner->children[0], inner->children[1]};
            const DTYPE oldLocal =
                node->WireToParent() + side->WireToParent() + inner->WireToParent()
                + inner->children[0]->WireToParent() + inner->children[1]->WireToParent();

            for (const auto& supportChild : nodes) {
                if (!supportChild->parent) continue;
                if (IsAncestorByTour(node, supportChild, tin, tout)) continue;
                auto supportParent = supportChild->parent;
                const auto supportPoints = EnumerateSupportPoints(supportChild, roots);
                for (const auto& supportPoint : supportPoints) {
                    if (HasDuplicateBundlePoints(supportPoint, roots)) continue;
                    const DTYPE fastLocalWl = FourPointRsmtLength(supportPoint, roots);
                    const DTYPE fastDelta = fastLocalWl - oldLocal;
                    if (fastDelta > fastMargin) continue;
                    if (!allowNeutral && fastDelta == 0 && fastMargin == 0) continue;
                    if (fastDelta > best.wireLengthDelta + fastMargin) continue;
                    auto local = SolveLocalFlute(supportPoint, roots);
                    if (!local.valid) continue;
                    const DTYPE delta = local.wl - oldLocal;

                    bool feasible = true;
                    const DTYPE basePath = pathLengths[supportParent->id] + Dist(supportParent->loc, supportPoint);
                    long long weightedPathDelta = 0;
                    for (size_t ridx = 0; ridx < roots.size(); ++ridx) {
                        const DTYPE pathDelta = basePath + local.sinkPathLengths[ridx] - pathLengths[roots[ridx]->id];
                        weightedPathDelta += static_cast<long long>(subtreePins[roots[ridx]->id]) * pathDelta;
                        if (pathDelta > slacks[roots[ridx]->id]) {
                            feasible = false;
                            break;
                        }
                    }
                    if (!feasible) continue;
                    if (delta > 0) continue;
                    if (!allowNeutral && delta == 0) continue;
                    if (allowNeutral && delta == 0 && weightedPathDelta >= 0) continue;
                    if (!BetterLexicographic(delta, weightedPathDelta, best.wireLengthDelta, best.weightedPathDelta)) continue;

                    best.wireLengthDelta = delta;
                    best.weightedPathDelta = weightedPathDelta;
                    best.bundleRoot = node;
                    best.innerChild = inner;
                    best.roots = roots;
                    best.supportChild = supportChild;
                    best.supportPoint = supportPoint;
                    best.localSource = local.tree.source;
                    local.tree.source.reset();
                    best.localPathLengths = std::move(local.sinkPathLengths);
                    best.valid = true;
                }
            }
        }
    }
    return best;
}

QuadMove FindBestQuadMove(Tree& tree, double eps, bool allowNeutral) {
    const int numNodes = tree.UpdateId();
    auto nodes = tree.ObtainNodes();
    vector<DTYPE> pathLengths(numNodes, 0);
    vector<DTYPE> slacks(numNodes, 0);
    vector<int> subtreePins(numNodes, 0);
    UpdatePathLengths(tree, pathLengths);
    UpdateSlacks(tree, pathLengths, slacks, eps);
    UpdateSubtreePinCounts(tree, subtreePins);
    vector<int> tin(static_cast<size_t>(numNodes), 0);
    vector<int> tout(static_cast<size_t>(numNodes), 0);
    int tourTimer = 0;
    UpdateEulerTour(tree.source, tin, tout, tourTimer);

    QuadMove best;
    for (const auto& node : nodes) {
        if (!node->parent || node->pin || node->children.size() != 2) continue;
        auto left = node->children[0];
        auto right = node->children[1];
        if (left->pin || right->pin) continue;
        if (left->children.size() != 2 || right->children.size() != 2) continue;

        vector<shared_ptr<TreeNode>> roots = {
            left->children[0], left->children[1], right->children[0], right->children[1]
        };
        const DTYPE oldLocal =
            node->WireToParent()
            + left->WireToParent() + right->WireToParent()
            + left->children[0]->WireToParent() + left->children[1]->WireToParent()
            + right->children[0]->WireToParent() + right->children[1]->WireToParent();

        for (const auto& supportChild : nodes) {
            if (!supportChild->parent) continue;
            if (IsAncestorByTour(node, supportChild, tin, tout)) continue;
            auto supportParent = supportChild->parent;
            const auto supportPoints = EnumerateSupportPoints(supportChild, roots);
            for (const auto& supportPoint : supportPoints) {
                auto local = SolveLocalFlute(supportPoint, roots);
                if (!local.valid) continue;
                const DTYPE delta = local.wl - oldLocal;

                bool feasible = true;
                const DTYPE basePath = pathLengths[supportParent->id] + Dist(supportParent->loc, supportPoint);
                long long weightedPathDelta = 0;
                for (size_t ridx = 0; ridx < roots.size(); ++ridx) {
                    const DTYPE pathDelta = basePath + local.sinkPathLengths[ridx] - pathLengths[roots[ridx]->id];
                    weightedPathDelta += static_cast<long long>(subtreePins[roots[ridx]->id]) * pathDelta;
                    if (pathDelta > slacks[roots[ridx]->id]) {
                        feasible = false;
                        break;
                    }
                }
                if (!feasible) continue;
                if (delta > 0) continue;
                if (!allowNeutral && delta == 0) continue;
                if (allowNeutral && delta == 0 && weightedPathDelta >= 0) continue;
                if (!BetterLexicographic(delta, weightedPathDelta, best.wireLengthDelta, best.weightedPathDelta)) continue;

                best.wireLengthDelta = delta;
                best.weightedPathDelta = weightedPathDelta;
                best.bundleRoot = node;
                best.innerChildren = {left, right};
                best.roots = roots;
                best.supportChild = supportChild;
                best.supportPoint = supportPoint;
                best.localSource = local.tree.source;
                local.tree.source.reset();
                best.localPathLengths = std::move(local.sinkPathLengths);
                best.valid = true;
            }
        }
    }
    return best;
}

PairMove FindBestSiblingPairMove(Tree& tree, double eps, bool allowNeutral) {
    const int numNodes = tree.UpdateId();
    auto nodes = tree.ObtainNodes();
    vector<DTYPE> pathLengths(numNodes, 0);
    vector<DTYPE> slacks(numNodes, 0);
    vector<int> subtreePins(numNodes, 0);
    UpdatePathLengths(tree, pathLengths);
    UpdateSlacks(tree, pathLengths, slacks, eps);
    UpdateSubtreePinCounts(tree, subtreePins);
    vector<int> tin(static_cast<size_t>(numNodes), 0);
    vector<int> tout(static_cast<size_t>(numNodes), 0);
    int tourTimer = 0;
    UpdateEulerTour(tree.source, tin, tout, tourTimer);

    PairMove bestMove;
    for (const auto& node : nodes) {
        if (!node->parent || node->pin || node->children.size() != 2) continue;

        auto left = node->children[0];
        auto right = node->children[1];
        const DTYPE oldLocalLength = node->WireToParent() + left->WireToParent() + right->WireToParent();

        for (const auto& supportChild : nodes) {
            if (!supportChild->parent) continue;
            if (IsAncestorByTour(node, supportChild, tin, tout)) continue;

            auto supportParent = supportChild->parent;
            const auto supportPoints = EnumerateSupportPoints(supportChild, left->loc, right->loc);
            for (const auto& supportPoint : supportPoints) {
                Point branchPoint;
                const DTYPE newLocalLength = ThreePointRsmtLength(left->loc, right->loc, supportPoint, branchPoint);
                const DTYPE wireLengthDelta = newLocalLength - oldLocalLength;

                const DTYPE basePath = pathLengths[supportParent->id] + Dist(supportParent->loc, supportPoint);
                const DTYPE newLeftPath = basePath + Dist(supportPoint, branchPoint) + Dist(branchPoint, left->loc);
                const DTYPE newRightPath = basePath + Dist(supportPoint, branchPoint) + Dist(branchPoint, right->loc);
                const DTYPE deltaLeft = newLeftPath - pathLengths[left->id];
                const DTYPE deltaRight = newRightPath - pathLengths[right->id];
                if (deltaLeft > slacks[left->id] || deltaRight > slacks[right->id]) continue;
                const long long weightedPathDelta =
                    static_cast<long long>(subtreePins[left->id]) * deltaLeft
                    + static_cast<long long>(subtreePins[right->id]) * deltaRight;
                if (wireLengthDelta > 0) continue;
                if (!allowNeutral && wireLengthDelta == 0) continue;
                if (allowNeutral && wireLengthDelta == 0 && weightedPathDelta >= 0) continue;
                if (!BetterLexicographic(
                        wireLengthDelta, weightedPathDelta, bestMove.wireLengthDelta, bestMove.weightedPathDelta))
                    continue;

                bestMove.wireLengthDelta = wireLengthDelta;
                bestMove.weightedPathDelta = weightedPathDelta;
                bestMove.bundleRoot = node;
                bestMove.left = left;
                bestMove.right = right;
                bestMove.supportChild = supportChild;
                bestMove.supportPoint = supportPoint;
                bestMove.branchPoint = branchPoint;
                bestMove.valid = true;
            }
        }
    }
    return bestMove;
}

void ApplyTripletMove(Tree& tree, const TripletMove& move) {
    TreeNode::ResetParent(move.roots[1]);
    TreeNode::ResetParent(move.roots[2]);
    TreeNode::ResetParent(move.roots[0]);
    TreeNode::ResetParent(move.innerChild);
    TreeNode::ResetParent(move.bundleRoot);

    auto attach = SplitSupportEdge(move.supportChild, move.supportPoint);
    MaterializeLocalFluteAtAttach(attach, move.localSource, move.roots);

    tree.RemovePhyRedundantSteiner();
    tree.RemoveTopoRedundantSteiner();
    tree.RemoveEmptyChildren();
}

void ApplyQuadMove(Tree& tree, const QuadMove& move) {
    for (const auto& root : move.roots) TreeNode::ResetParent(root);
    for (const auto& child : move.innerChildren) TreeNode::ResetParent(child);
    TreeNode::ResetParent(move.bundleRoot);

    auto attach = SplitSupportEdge(move.supportChild, move.supportPoint);
    MaterializeLocalFluteAtAttach(attach, move.localSource, move.roots);

    tree.RemovePhyRedundantSteiner();
    tree.RemoveTopoRedundantSteiner();
    tree.RemoveEmptyChildren();
}

void ApplySiblingPairMove(Tree& tree, const PairMove& move) {
    TreeNode::ResetParent(move.left);
    TreeNode::ResetParent(move.right);
    TreeNode::ResetParent(move.bundleRoot);

    auto attach = SplitSupportEdge(move.supportChild, move.supportPoint);
    MaterializeThreePointConnector(attach, move.left, move.right, move.branchPoint);

    tree.RemovePhyRedundantSteiner();
    tree.RemoveTopoRedundantSteiner();
    tree.RemoveEmptyChildren();
}

}  // namespace

void Refine::TripletBundleSubstitute(Tree& tree, double eps, bool allowNeutral) {
    const int maxRounds = std::max(0, RefineEnvInt("SALT_TRIPLET_BUNDLE_MAX_ROUNDS", 0));
    int rounds = 0;
    while (maxRounds <= 0 || rounds < maxRounds) {
        const bool verifyMove = allowNeutral || RefineEnvInt("SALT_BUNDLE_VERIFY", 0) != 0;
        const DTYPE wlBefore = verifyMove ? WireLengthEvalBase(tree).wireLength : 0;
        const long long totalPathBefore = allowNeutral ? TotalPinPathLength(tree) : 0;

        const int numNodes = tree.UpdateId();
        auto nodes = tree.ObtainNodes();
        vector<DTYPE> pathLengths(numNodes, 0);
        vector<DTYPE> slacks(numNodes, 0);
        vector<int> subtreePins(numNodes, 0);
        UpdatePathLengths(tree, pathLengths);
        UpdateSlacks(tree, pathLengths, slacks, eps);
        UpdateSubtreePinCounts(tree, subtreePins);
        vector<int> tin(static_cast<size_t>(numNodes), 0);
        vector<int> tout(static_cast<size_t>(numNodes), 0);
        int tourTimer = 0;
        UpdateEulerTour(tree.source, tin, tout, tourTimer);

        TripletMove best;
        const int fastMargin = std::max(0, RefineEnvInt("SALT_TRIPLET_FAST4_MARGIN", 0));
        const bool edgeHpwlPrefilter = RefineEnvInt("SALT_TRIPLET_EDGE_HPWL_PREFILTER", 0) != 0;
        const BundleLocalityLimits localityLimits = MakeBundleLocalityLimits("SALT_TRIPLET_BUNDLE");
        vector<int> depths;
        if (localityLimits.enabled) {
            depths.assign(static_cast<size_t>(numNodes), 0);
            UpdateDepths(tree, depths);
        }
        const vector<unsigned char> lcaLocalityAllowed =
            BuildBundleLcaLocalityMatrix(nodes, numNodes, depths, localityLimits);
        const vector<vector<shared_ptr<TreeNode>>> lcaSupportLists =
            BuildBundleLcaSupportLists(nodes, nodes, numNodes, depths, localityLimits);

        const int rootCap = BundleRootCapFor("SALT_TRIPLET_BUNDLE");
        vector<unsigned char> rootAllowed;
        if (rootCap > 0) {
            rootAllowed.assign(static_cast<size_t>(numNodes) * 2, 0);
            vector<BundleRootPotential> potentials;
            int order = 0;
            for (const auto& node : nodes) {
                if (!node->parent || node->pin || node->children.size() != 2) continue;
                for (int innerIdx = 0; innerIdx < 2; ++innerIdx) {
                    auto inner = node->children[static_cast<size_t>(innerIdx)];
                    auto side = node->children[static_cast<size_t>(1 - innerIdx)];
                    if (inner->pin || inner->children.size() != 2) continue;
                    vector<shared_ptr<TreeNode>> roots = {side, inner->children[0], inner->children[1]};
                    const DTYPE oldLocal =
                        node->WireToParent() + side->WireToParent() + inner->WireToParent()
                        + inner->children[0]->WireToParent() + inner->children[1]->WireToParent();
                    potentials.push_back(
                        {node->id, innerIdx, oldLocal - BundleBoundaryHpwlLowerBound(roots), order++});
                }
            }
            MarkTopBundleRootPotentials(rootAllowed, 2, potentials, rootCap);
        }

        for (const auto& node : nodes) {
            if (!node->parent || node->pin || node->children.size() != 2) continue;
            for (int innerIdx = 0; innerIdx < 2; ++innerIdx) {
                auto inner = node->children[static_cast<size_t>(innerIdx)];
                auto side = node->children[static_cast<size_t>(1 - innerIdx)];
                if (inner->pin || inner->children.size() != 2) continue;
                if (!rootAllowed.empty() &&
                    !rootAllowed[static_cast<size_t>(node->id * 2 + innerIdx)]) {
                    continue;
                }

                vector<shared_ptr<TreeNode>> roots = {side, inner->children[0], inner->children[1]};
                const DTYPE oldLocal =
                    node->WireToParent() + side->WireToParent() + inner->WireToParent()
                    + inner->children[0]->WireToParent() + inner->children[1]->WireToParent();

                const auto& supportCandidates = BundleSupportCandidatesFor(node, nodes, lcaSupportLists);
                for (const auto& supportChild : supportCandidates) {
                    if (!supportChild->parent) continue;
                    if (IsAncestorByTour(node, supportChild, tin, tout)) continue;
                    if (!BundleSupportEdgeAllowed(
                            node, supportChild, pathLengths, depths, localityLimits,
                            &lcaLocalityAllowed, numNodes)) {
                        continue;
                    }

                    auto supportParent = supportChild->parent;
                    if (edgeHpwlPrefilter) {
                        const DTYPE hpwlDelta = BundleHpwlLowerBoundOnSupportEdge(supportChild, roots) - oldLocal;
                        if (hpwlDelta > fastMargin || (!allowNeutral && hpwlDelta == 0 && fastMargin == 0)) {
                            continue;
                        }
                    }

                    const auto supportPoints = EnumerateSupportPoints(supportChild, roots);
                    for (const auto& supportPoint : supportPoints) {
                        if (HasDuplicateBundlePoints(supportPoint, roots)) continue;
                        const DTYPE fastLocalWl = FourPointRsmtLength(supportPoint, roots);
                        const DTYPE fastDelta = fastLocalWl - oldLocal;
                        if (fastDelta > fastMargin) continue;
                        if (!allowNeutral && fastDelta == 0 && fastMargin == 0) continue;
                        if (fastDelta > best.wireLengthDelta + fastMargin) continue;

                        auto local = SolveLocalFlute(supportPoint, roots);
                        if (!local.valid) continue;
                        const DTYPE delta = local.wl - oldLocal;
                        if (delta > 0) continue;
                        if (!allowNeutral && delta == 0) continue;

                        bool feasible = true;
                        const DTYPE basePath = pathLengths[supportParent->id] + Dist(supportParent->loc, supportPoint);
                        long long weightedPathDelta = 0;
                        for (size_t ridx = 0; ridx < roots.size(); ++ridx) {
                            const DTYPE pathDelta =
                                basePath + local.sinkPathLengths[ridx] - pathLengths[roots[ridx]->id];
                            weightedPathDelta += static_cast<long long>(subtreePins[roots[ridx]->id]) * pathDelta;
                            if (pathDelta > slacks[roots[ridx]->id]) {
                                feasible = false;
                                break;
                            }
                        }
                        if (!feasible) continue;
                        if (allowNeutral && delta == 0 && weightedPathDelta >= 0) continue;
                        if (!BetterLexicographic(
                                delta, weightedPathDelta, best.wireLengthDelta, best.weightedPathDelta)) {
                            continue;
                        }

                        best.wireLengthDelta = delta;
                        best.weightedPathDelta = weightedPathDelta;
                        best.bundleRoot = node;
                        best.innerChild = inner;
                        best.roots = roots;
                        best.supportChild = supportChild;
                        best.supportPoint = supportPoint;
                        best.localSource = local.tree.source;
                        local.tree.source.reset();
                        best.localPathLengths = std::move(local.sinkPathLengths);
                        best.valid = true;
                    }
                }
            }
        }

        if (!best.valid) break;
        ApplyTripletMove(tree, best);
        if (verifyMove) {
            tree.QuickCheck();
            const DTYPE wlAfter = WireLengthEvalBase(tree).wireLength;
            if (allowNeutral && wlAfter == wlBefore) {
                const long long totalPathAfter = TotalPinPathLength(tree);
                assert(totalPathAfter < totalPathBefore);
            } else {
                assert(wlAfter < wlBefore);
            }
        }
        ++rounds;
    }
}

void Refine::QuadBundleSubstitute(Tree& tree, double eps, bool allowNeutral) {
    const int maxRounds = std::max(0, RefineEnvInt("SALT_QUAD_BUNDLE_MAX_ROUNDS", 0));
    int rounds = 0;
    while (maxRounds <= 0 || rounds < maxRounds) {
        const bool verifyMove = allowNeutral || RefineEnvInt("SALT_BUNDLE_VERIFY", 0) != 0;
        const DTYPE wlBefore = verifyMove ? WireLengthEvalBase(tree).wireLength : 0;
        const long long totalPathBefore = allowNeutral ? TotalPinPathLength(tree) : 0;

        const int numNodes = tree.UpdateId();
        auto nodes = tree.ObtainNodes();
        vector<DTYPE> pathLengths(numNodes, 0);
        vector<DTYPE> slacks(numNodes, 0);
        vector<int> subtreePins(numNodes, 0);
        UpdatePathLengths(tree, pathLengths);
        UpdateSlacks(tree, pathLengths, slacks, eps);
        UpdateSubtreePinCounts(tree, subtreePins);
        vector<int> tin(static_cast<size_t>(numNodes), 0);
        vector<int> tout(static_cast<size_t>(numNodes), 0);
        int tourTimer = 0;
        UpdateEulerTour(tree.source, tin, tout, tourTimer);

        QuadMove best;
        const bool hpwlPrefilter = RefineEnvInt("SALT_QUAD_HPWL_PREFILTER", 0) != 0;
        const bool edgeHpwlPrefilter = RefineEnvInt("SALT_QUAD_EDGE_HPWL_PREFILTER", 0) != 0;
        const BundleLocalityLimits localityLimits = MakeBundleLocalityLimits("SALT_QUAD_BUNDLE");
        vector<int> depths;
        if (localityLimits.enabled) {
            depths.assign(static_cast<size_t>(numNodes), 0);
            UpdateDepths(tree, depths);
        }
        const vector<unsigned char> lcaLocalityAllowed =
            BuildBundleLcaLocalityMatrix(nodes, numNodes, depths, localityLimits);
        const vector<vector<shared_ptr<TreeNode>>> lcaSupportLists =
            BuildBundleLcaSupportLists(nodes, nodes, numNodes, depths, localityLimits);

        const int rootCap = BundleRootCapFor("SALT_QUAD_BUNDLE");
        vector<unsigned char> rootAllowed;
        if (rootCap > 0) {
            rootAllowed.assign(static_cast<size_t>(numNodes), 0);
            vector<BundleRootPotential> potentials;
            int order = 0;
            for (const auto& node : nodes) {
                if (!node->parent || node->pin || node->children.size() != 2) continue;
                auto left = node->children[0];
                auto right = node->children[1];
                if (left->pin || right->pin) continue;
                if (left->children.size() != 2 || right->children.size() != 2) continue;
                vector<shared_ptr<TreeNode>> roots = {
                    left->children[0], left->children[1], right->children[0], right->children[1]};
                const DTYPE oldLocal =
                    node->WireToParent()
                    + left->WireToParent() + right->WireToParent()
                    + left->children[0]->WireToParent() + left->children[1]->WireToParent()
                    + right->children[0]->WireToParent() + right->children[1]->WireToParent();
                potentials.push_back({node->id, 0, oldLocal - BundleBoundaryHpwlLowerBound(roots), order++});
            }
            MarkTopBundleRootPotentials(rootAllowed, 1, potentials, rootCap);
        }

        for (const auto& node : nodes) {
            if (!node->parent || node->pin || node->children.size() != 2) continue;
            auto left = node->children[0];
            auto right = node->children[1];
            if (left->pin || right->pin) continue;
            if (left->children.size() != 2 || right->children.size() != 2) continue;
            if (!rootAllowed.empty() && !rootAllowed[static_cast<size_t>(node->id)]) continue;

            vector<shared_ptr<TreeNode>> roots = {
                left->children[0], left->children[1], right->children[0], right->children[1]};
            const DTYPE oldLocal =
                node->WireToParent()
                + left->WireToParent() + right->WireToParent()
                + left->children[0]->WireToParent() + left->children[1]->WireToParent()
                + right->children[0]->WireToParent() + right->children[1]->WireToParent();

            const auto& supportCandidates = BundleSupportCandidatesFor(node, nodes, lcaSupportLists);
            for (const auto& supportChild : supportCandidates) {
                if (!supportChild->parent) continue;
                if (IsAncestorByTour(node, supportChild, tin, tout)) continue;
                if (!BundleSupportEdgeAllowed(
                        node, supportChild, pathLengths, depths, localityLimits,
                        &lcaLocalityAllowed, numNodes)) {
                    continue;
                }

                auto supportParent = supportChild->parent;
                if (edgeHpwlPrefilter) {
                    const DTYPE hpwlDelta = BundleHpwlLowerBoundOnSupportEdge(supportChild, roots) - oldLocal;
                    if (hpwlDelta > 0 || (!allowNeutral && hpwlDelta == 0)) continue;
                }

                const auto supportPoints = EnumerateSupportPoints(supportChild, roots);
                for (const auto& supportPoint : supportPoints) {
                    if (hpwlPrefilter) {
                        const DTYPE hpwlDelta = BundleHpwlLowerBound(supportPoint, roots) - oldLocal;
                        if (hpwlDelta > 0 || (!allowNeutral && hpwlDelta == 0)) continue;
                    }

                    auto local = SolveLocalFlute(supportPoint, roots);
                    if (!local.valid) continue;
                    const DTYPE delta = local.wl - oldLocal;
                    if (delta > 0) continue;
                    if (!allowNeutral && delta == 0) continue;

                    bool feasible = true;
                    const DTYPE basePath = pathLengths[supportParent->id] + Dist(supportParent->loc, supportPoint);
                    long long weightedPathDelta = 0;
                    for (size_t ridx = 0; ridx < roots.size(); ++ridx) {
                        const DTYPE pathDelta =
                            basePath + local.sinkPathLengths[ridx] - pathLengths[roots[ridx]->id];
                        weightedPathDelta += static_cast<long long>(subtreePins[roots[ridx]->id]) * pathDelta;
                        if (pathDelta > slacks[roots[ridx]->id]) {
                            feasible = false;
                            break;
                        }
                    }
                    if (!feasible) continue;
                    if (allowNeutral && delta == 0 && weightedPathDelta >= 0) continue;
                    if (!BetterLexicographic(delta, weightedPathDelta, best.wireLengthDelta, best.weightedPathDelta)) {
                        continue;
                    }

                    best.wireLengthDelta = delta;
                    best.weightedPathDelta = weightedPathDelta;
                    best.bundleRoot = node;
                    best.innerChildren = {left, right};
                    best.roots = roots;
                    best.supportChild = supportChild;
                    best.supportPoint = supportPoint;
                    best.localSource = local.tree.source;
                    local.tree.source.reset();
                    best.localPathLengths = std::move(local.sinkPathLengths);
                    best.valid = true;
                }
            }
        }

        if (!best.valid) break;
        ApplyQuadMove(tree, best);
        if (verifyMove) {
            tree.QuickCheck();
            const DTYPE wlAfter = WireLengthEvalBase(tree).wireLength;
            if (allowNeutral && wlAfter == wlBefore) {
                const long long totalPathAfter = TotalPinPathLength(tree);
                assert(totalPathAfter < totalPathBefore);
            } else {
                assert(wlAfter < wlBefore);
            }
        }
        ++rounds;
    }
}

void Refine::SiblingPairSubstitute(Tree& tree, double eps, bool /*useRTree*/, bool allowNeutral) {
    const int maxRounds = std::max(0, RefineEnvInt("SALT_SIBLING_PAIR_MAX_ROUNDS", 0));
    int rounds = 0;
    while (maxRounds <= 0 || rounds < maxRounds) {
        const bool verifyMove = allowNeutral || RefineEnvInt("SALT_BUNDLE_VERIFY", 0) != 0;
        const DTYPE wlBefore = verifyMove ? WireLengthEvalBase(tree).wireLength : 0;
        const long long totalPathBefore = allowNeutral ? TotalPinPathLength(tree) : 0;

        const int numNodes = tree.UpdateId();
        auto nodes = tree.ObtainNodes();
        vector<DTYPE> pathLengths(numNodes, 0);
        vector<DTYPE> slacks(numNodes, 0);
        vector<int> subtreePins(numNodes, 0);
        UpdatePathLengths(tree, pathLengths);
        UpdateSlacks(tree, pathLengths, slacks, eps);
        UpdateSubtreePinCounts(tree, subtreePins);
        vector<int> tin(static_cast<size_t>(numNodes), 0);
        vector<int> tout(static_cast<size_t>(numNodes), 0);
        int tourTimer = 0;
        UpdateEulerTour(tree.source, tin, tout, tourTimer);

        const bool edgeMinPrefilter = RefineEnvInt("SALT_SIBLING_PAIR_EDGE_MIN_PREFILTER", 0) != 0;
        const BundleLocalityLimits localityLimits = MakeBundleLocalityLimits("SALT_SIBLING_PAIR");
        vector<int> depths;
        if (localityLimits.enabled) {
            depths.assign(static_cast<size_t>(numNodes), 0);
            UpdateDepths(tree, depths);
        }
        const vector<unsigned char> lcaLocalityAllowed =
            BuildBundleLcaLocalityMatrix(nodes, numNodes, depths, localityLimits);
        const vector<vector<shared_ptr<TreeNode>>> lcaSupportLists =
            BuildBundleLcaSupportLists(nodes, nodes, numNodes, depths, localityLimits);
        const int rootCap = BundleRootCapFor("SALT_SIBLING_PAIR");
        vector<unsigned char> rootAllowed;
        if (rootCap > 0) {
            rootAllowed.assign(static_cast<size_t>(numNodes), 0);
            vector<BundleRootPotential> potentials;
            int order = 0;
            for (const auto& node : nodes) {
                if (!node->parent || node->pin || node->children.size() != 2) continue;
                auto left = node->children[0];
                auto right = node->children[1];
                const DTYPE oldLocalLength = node->WireToParent() + left->WireToParent() + right->WireToParent();
                const DTYPE lb = std::abs(left->loc.x - right->loc.x) + std::abs(left->loc.y - right->loc.y);
                potentials.push_back({node->id, 0, oldLocalLength - lb, order++});
            }
            MarkTopBundleRootPotentials(rootAllowed, 1, potentials, rootCap);
        }

        PairMove bestMove;
        for (const auto& node : nodes) {
            if (!node->parent || node->pin || node->children.size() != 2) continue;
            if (!rootAllowed.empty() && !rootAllowed[static_cast<size_t>(node->id)]) continue;

            auto left = node->children[0];
            auto right = node->children[1];
            const DTYPE oldLocalLength = node->WireToParent() + left->WireToParent() + right->WireToParent();

            const auto& supportCandidates = BundleSupportCandidatesFor(node, nodes, lcaSupportLists);
            for (const auto& supportChild : supportCandidates) {
                if (!supportChild->parent) continue;
                if (IsAncestorByTour(node, supportChild, tin, tout)) continue;

                if (!BundleSupportEdgeAllowed(
                        node, supportChild, pathLengths, depths, localityLimits,
                        &lcaLocalityAllowed, numNodes)) {
                    continue;
                }
                auto supportParent = supportChild->parent;
                if (edgeMinPrefilter) {
                    const DTYPE minLocalLength =
                        ThreePointRsmtMinLengthOnSupportEdge(supportChild, left->loc, right->loc);
                    const DTYPE minDelta = minLocalLength - oldLocalLength;
                    if (minDelta > bestMove.wireLengthDelta ||
                        (!bestMove.valid && !allowNeutral && minDelta == 0)) {
                        continue;
                    }
                }
                const auto supportPoints = EnumerateSupportPoints(supportChild, left->loc, right->loc);
                for (const auto& supportPoint : supportPoints) {
                    Point branchPoint;
                    const DTYPE newLocalLength = ThreePointRsmtLength(left->loc, right->loc, supportPoint, branchPoint);
                    const DTYPE wireLengthDelta = newLocalLength - oldLocalLength;
                    if (wireLengthDelta > 0) continue;
                    if (!allowNeutral && wireLengthDelta == 0) continue;

                    const DTYPE basePath = pathLengths[supportParent->id] + Dist(supportParent->loc, supportPoint);
                    const DTYPE newLeftPath = basePath + Dist(supportPoint, branchPoint) + Dist(branchPoint, left->loc);
                    const DTYPE newRightPath = basePath + Dist(supportPoint, branchPoint) + Dist(branchPoint, right->loc);
                    const DTYPE deltaLeft = newLeftPath - pathLengths[left->id];
                    const DTYPE deltaRight = newRightPath - pathLengths[right->id];
                    if (deltaLeft > slacks[left->id] || deltaRight > slacks[right->id]) continue;
                    const long long weightedPathDelta =
                        static_cast<long long>(subtreePins[left->id]) * deltaLeft
                        + static_cast<long long>(subtreePins[right->id]) * deltaRight;
                    if (allowNeutral && wireLengthDelta == 0 && weightedPathDelta >= 0) continue;
                    if (!BetterLexicographic(
                            wireLengthDelta, weightedPathDelta, bestMove.wireLengthDelta, bestMove.weightedPathDelta))
                        continue;

                    bestMove.wireLengthDelta = wireLengthDelta;
                    bestMove.weightedPathDelta = weightedPathDelta;
                    bestMove.bundleRoot = node;
                    bestMove.left = left;
                    bestMove.right = right;
                    bestMove.supportChild = supportChild;
                    bestMove.supportPoint = supportPoint;
                    bestMove.branchPoint = branchPoint;
                    bestMove.valid = true;
                }
            }
        }

        if (!bestMove.valid) break;

        ApplySiblingPairMove(tree, bestMove);
        if (verifyMove) {
            tree.QuickCheck();
            const DTYPE wlAfter = WireLengthEvalBase(tree).wireLength;
            if (allowNeutral && wlAfter == wlBefore) {
                const long long totalPathAfter = TotalPinPathLength(tree);
                assert(totalPathAfter < totalPathBefore);
            } else {
                assert(wlAfter < wlBefore);
            }
        }
        ++rounds;
    }
}

}  // namespace salt
