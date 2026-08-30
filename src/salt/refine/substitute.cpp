#include "refine.h"

#include "salt/base/eval.h"
#include "salt/base/mst.h"

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/segment.hpp>
#include <boost/geometry/index/rtree.hpp>

#include <algorithm>
#include <cstdlib>

namespace salt {

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

using BPoint = bg::model::point<DTYPE, 2, bg::cs::cartesian>;
using BSegment = bg::model::segment<BPoint>;
using BBox = bg::model::box<BPoint>;
using BPolygon = bg::model::polygon<BPoint>;
using RNode = pair<BBox, shared_ptr<TreeNode>>;  // R-Tree node
struct RNodeComp {
    bool operator()(const RNode& l, const RNode& r) const {
        return bg::equals(l.first, r.first) && l.second == r.second;
    }
};
using RTree = bgi::rtree<RNode, bgi::rstar<8>, bgi::indexable<RNode>, RNodeComp>;

BBox EdgeBox(const Point& a, const Point& b) {
    BBox s;
    bg::envelope(BSegment(BPoint(a.x, a.y), BPoint(b.x, b.y)), s);
    return s;
}

struct LinearEdge {
    DTYPE min_x = 0;
    DTYPE max_x = 0;
    DTYPE min_y = 0;
    DTYPE max_y = 0;
    shared_ptr<TreeNode> node;
};

LinearEdge MakeLinearEdge(const shared_ptr<TreeNode>& n, const shared_ptr<TreeNode>& parent) {
    return {std::min(n->loc.x, parent->loc.x), std::max(n->loc.x, parent->loc.x),
            std::min(n->loc.y, parent->loc.y), std::max(n->loc.y, parent->loc.y), n};
}

bool IntersectsBox(const LinearEdge& edge, DTYPE min_x, DTYPE max_x, DTYPE min_y, DTYPE max_y) {
    return edge.min_x <= max_x && min_x <= edge.max_x && edge.min_y <= max_y && min_y <= edge.max_y;
}

void Refine::Substitute(Tree& tree, double eps, bool useRTree, int maxRounds,
                        int linearCandidateCap, int linearQueryMode) {
    const bool useLinearIndex = useRTree && std::getenv("SALT_SUBSTITUTE_LINEAR_INDEX") != nullptr;
    const bool useLinearSortedQuery = useLinearIndex && linearQueryMode == 1;
    linearCandidateCap = std::max(0, linearCandidateCap);
    const int maxSubstituteRounds = std::max(0, maxRounds);
    RTree rtree;
    vector<LinearEdge> linearEdges;
    vector<int> linearEdgesByMinX;
    bool linearEdgesByMinXValid = false;
    if (useLinearIndex && tree.net) linearEdges.reserve(tree.net->pins.size() * 4);
    auto InsertIndexEdge = [&](const shared_ptr<TreeNode>& n, const shared_ptr<TreeNode>& parent) {
        if (useLinearIndex) {
            linearEdges.push_back(MakeLinearEdge(n, parent));
            linearEdgesByMinXValid = false;
        } else {
            rtree.insert({EdgeBox(n->loc, parent->loc), n});
        }
    };
    auto RemoveIndexEdge = [&](const shared_ptr<TreeNode>& n, const shared_ptr<TreeNode>& parent) {
        if (useLinearIndex) {
            (void)parent;
            for (int i = static_cast<int>(linearEdges.size()) - 1; i >= 0; --i) {
                if (linearEdges[static_cast<size_t>(i)].node == n) {
                    linearEdges[static_cast<size_t>(i)] = std::move(linearEdges.back());
                    linearEdges.pop_back();
                    linearEdgesByMinXValid = false;
                    break;
                }
            }
        } else {
            rtree.remove({EdgeBox(n->loc, parent->loc), n});
        }
    };
    if (useRTree) {
        auto insertEdges = [&](auto&& self, const shared_ptr<TreeNode>& n) -> void {
            for (const auto& child : n->children) self(self, child);
            if (n->parent) {
                InsertIndexEdge(n, n->parent);
            }
        };
        if (tree.source) insertEdges(insertEdges, tree.source);
    }
    auto Disconnect = [&](const shared_ptr<TreeNode>& n) {
        if (useRTree) {
            RemoveIndexEdge(n, n->parent);
        }
        TreeNode::ResetParent(n);
    };
    auto Connect = [&](const shared_ptr<TreeNode>& n, const shared_ptr<TreeNode>& parent) {
        TreeNode::SetParent(n, parent);
        if (useRTree) {
            InsertIndexEdge(n, parent);
        }
    };
    int substituteRounds = 0;
    while (maxSubstituteRounds <= 0 || substituteRounds < maxSubstituteRounds) {
        // Get nearest neighbors
        int num = tree.net ? static_cast<int>(tree.net->pins.size()) : 0;
        auto updateIds = [&](auto&& self, const shared_ptr<TreeNode>& node) -> void {
            if (node->pin) {
                node->id = node->pin->id;
            } else {
                node->id = num++;
            }
            for (const auto& child : node->children) self(self, child);
        };
        updateIds(updateIds, tree.source);
        vector<shared_ptr<TreeNode>> nodes(static_cast<size_t>(num));  // note: all pins should be covered
        vector<Point> points;
        if (!useRTree) points.resize(static_cast<size_t>(num));
        vector<int> tin;
        vector<int> tout;
        int linearPreOrderIdx = 0;
        if (useLinearIndex) {
            tin.resize(static_cast<size_t>(num));
            tout.resize(static_cast<size_t>(num));
        }
        auto collectNodes = [&](auto&& self, const shared_ptr<TreeNode>& node) -> void {
            nodes[static_cast<size_t>(node->id)] = node;
            if (!useRTree) points[static_cast<size_t>(node->id)] = node->loc;
            if (useLinearIndex) tin[static_cast<size_t>(node->id)] = linearPreOrderIdx++;
            for (const auto& child : node->children) self(self, child);
            if (useLinearIndex) tout[static_cast<size_t>(node->id)] = linearPreOrderIdx - 1;
        };
        collectNodes(collectNodes, tree.source);
        vector<vector<int>> nearestNeighbors;
        if (!useRTree) {
            MstBuilder mstB;
            mstB.GetAllNearestNeighbors(points, nearestNeighbors);
        } else if (!useLinearIndex) {
            nearestNeighbors.resize(nodes.size());
            for (const auto& n : nodes) {
                if (n->parent) {
                    Point c = n->loc;  // center
                    DTYPE radius = n->WireToParent();
                    // diamond is too slow...
                    // BPolygon diamond;
                    // diamond.outer().emplace_back(c.x - radius, c.y);
                    // diamond.outer().emplace_back(c.x, c.y + radius);
                    // diamond.outer().emplace_back(c.x + radius, c.y);
                    // diamond.outer().emplace_back(c.x, c.y - radius);
                    // diamond.outer().emplace_back(c.x - radius, c.y);
                    auto& neighs = nearestNeighbors[n->id];
                    BBox queryBox{{c.x - radius, c.y - radius}, {c.x + radius, c.y + radius}};
                    for (auto it = rtree.qbegin(bgi::intersects(queryBox)); it != rtree.qend(); ++it) {
                        neighs.push_back(it->second->id);
                    }
                }
            }
        }

        // Prune descendants in nearest neighbors.  The linear index fast path uses tin/tout directly.
        if (!useLinearIndex) {
            vector<int> preOrderIdxes(nodes.size(), -1);
            int globalPreOrderIdx = 0;
            auto removeDescendants = [&](auto&& self, const shared_ptr<TreeNode>& node) -> void {
                preOrderIdxes[node->id] = globalPreOrderIdx++;
                for (const auto& child : node->children) {
                    self(self, child);
                }
                for (auto& neighIdx : nearestNeighbors[node->id]) {
                    int neighPreOrderIdx = preOrderIdxes[neighIdx];
                    if (neighPreOrderIdx != -1 && neighPreOrderIdx >= preOrderIdxes[node->id]) {
                        neighIdx = -1;  // -1 stands for "descendant"
                    }
                }
            };
            removeDescendants(removeDescendants, tree.source);
        }

        // Init path lengths and subtree slacks
        vector<DTYPE> pathLengths(nodes.size());
        vector<DTYPE> slacks(nodes.size());
        auto UpdatePathLengths = [&](const shared_ptr<TreeNode>& node) {
            if (node->parent) {
                pathLengths[node->id] = pathLengths[node->parent->id] + node->WireToParent();
            } else {
                pathLengths[node->id] = 0;
            }
        };
        auto UpdateSlacks = [&](const shared_ptr<TreeNode>& node) {
            if (node->children.empty()) {
                slacks[node->id] =
                    Dist(node->loc, tree.source->loc) * (1 + eps) - pathLengths[node->id];  // floor here...
            } else {
                DTYPE minSlack = Dist(node->loc, tree.source->loc) * (1 + eps) - pathLengths[node->id];
                for (const auto& child : node->children) {
                    minSlack = min(minSlack, slacks[child->id]);
                }
                slacks[node->id] = minSlack;
            }
        };
        auto updatePathLengthsDfs = [&](auto&& self, const shared_ptr<TreeNode>& node) -> void {
            UpdatePathLengths(node);
            for (const auto& child : node->children) self(self, child);
        };
        auto updateSlacksDfs = [&](auto&& self, const shared_ptr<TreeNode>& node) -> void {
            for (const auto& child : node->children) self(self, child);
            UpdateSlacks(node);
        };
        updatePathLengthsDfs(updatePathLengthsDfs, tree.source);
        updateSlacksDfs(updateSlacksDfs, tree.source);

        // Find legal candidate moves
        using MoveT = tuple<DTYPE, int, int>;
        vector<MoveT> candidateMoves;  // <wireLengthDelta, node id, new parent id>
        auto GetNearestPoint = [](const shared_ptr<TreeNode>& target, const shared_ptr<TreeNode>& neigh) {
            Box box(neigh->loc, neigh->parent->loc);
            box.Legalize();
            return box.GetNearestPointTo(target->loc);
        };
        for (const auto& node : nodes) {
            if (!(node->parent)) {
                continue;
            }
            DTYPE bestWireLengthDelta = 0;  // the negative, the better
            int bestNewParentId = -1;
            auto considerNeighbor = [&](const shared_ptr<TreeNode>& neigh) {
                if (!neigh || !neigh->parent) return;
                const int neighIdx = neigh->id;
                if (neighIdx < 0 || neighIdx >= static_cast<int>(tin.size()) ||
                    node->id < 0 || node->id >= static_cast<int>(tin.size())) {
                    return;
                }
                if (useLinearIndex && tin[static_cast<size_t>(node->id)] <= tin[static_cast<size_t>(neighIdx)] &&
                    tin[static_cast<size_t>(neighIdx)] <= tout[static_cast<size_t>(node->id)]) {
                    return;
                }
                auto neighParent = neigh->parent;
                auto steinerPt = GetNearestPoint(node, neigh);
                DTYPE wireLengthDelta = Dist(node->loc, steinerPt) - node->WireToParent();
                if (wireLengthDelta < bestWireLengthDelta) {  // has wire length improvement
                    DTYPE pathLengthDelta =
                        pathLengths[neighParent->id] + Dist(node->loc, neighParent->loc) - pathLengths[node->id];
                    if (pathLengthDelta <= slacks[node->id]) {  // make path length under control
                        bestWireLengthDelta = wireLengthDelta;
                        bestNewParentId = neigh->id;
                    }
                }
            };
            if (useLinearIndex) {
                const Point c = node->loc;  // center
                const DTYPE radius = node->WireToParent();
                const DTYPE min_x = c.x - radius;
                const DTYPE max_x = c.x + radius;
                const DTYPE min_y = c.y - radius;
                const DTYPE max_y = c.y + radius;
                int checkedCandidates = 0;
                if (useLinearSortedQuery && !linearEdgesByMinXValid) {
                    linearEdgesByMinX.resize(linearEdges.size());
                    for (int i = 0; i < static_cast<int>(linearEdges.size()); ++i) {
                        linearEdgesByMinX[static_cast<size_t>(i)] = i;
                    }
                    std::sort(linearEdgesByMinX.begin(), linearEdgesByMinX.end(),
                              [&](int lhs, int rhs) {
                                  const auto& l = linearEdges[static_cast<size_t>(lhs)];
                                  const auto& r = linearEdges[static_cast<size_t>(rhs)];
                                  if (l.min_x != r.min_x) return l.min_x < r.min_x;
                                  return lhs < rhs;
                              });
                    linearEdgesByMinXValid = true;
                }
                auto handle_edge = [&](const LinearEdge& edge) {
                    if (!IntersectsBox(edge, min_x, max_x, min_y, max_y)) return false;
                    considerNeighbor(edge.node);
                    ++checkedCandidates;
                    return linearCandidateCap > 0 && checkedCandidates >= linearCandidateCap;
                };
                if (useLinearSortedQuery) {
                    vector<int> hits;
                    auto stopIt = std::upper_bound(
                        linearEdgesByMinX.begin(), linearEdgesByMinX.end(), max_x,
                        [&](DTYPE value, int edgeId) {
                            return value < linearEdges[static_cast<size_t>(edgeId)].min_x;
                        });
                    hits.reserve(static_cast<size_t>(std::distance(linearEdgesByMinX.begin(), stopIt)));
                    for (auto it = linearEdgesByMinX.begin(); it != stopIt; ++it) {
                        const int edgeId = *it;
                        const auto& edge = linearEdges[static_cast<size_t>(edgeId)];
                        if (edge.max_x < min_x || edge.min_y > max_y || edge.max_y < min_y) continue;
                        hits.push_back(edgeId);
                    }
                    std::sort(hits.begin(), hits.end());
                    for (int edgeId : hits) {
                        if (handle_edge(linearEdges[static_cast<size_t>(edgeId)])) break;
                    }
                } else {
                    for (const auto& edge : linearEdges) {
                        if (handle_edge(edge)) break;
                    }
                }
            } else {
                for (int neighIdx : nearestNeighbors[node->id]) {
                    if (neighIdx == -1) continue;
                    considerNeighbor(nodes[static_cast<size_t>(neighIdx)]);
                }
            }
            if (bestNewParentId >= 0) {
                candidateMoves.emplace_back(bestWireLengthDelta, node->id, bestNewParentId);
            }
        }
        if (candidateMoves.empty()) {
            break;
        }

        // Try candidate moves in the order of descending wire length savings
        // Note that earlier moves may influence the legality of later one
        sort(candidateMoves.begin(), candidateMoves.end(), [](const MoveT& lhs, const MoveT& rhs){
            return get<0>(lhs) < get<0>(rhs);
        });
        for (const auto& move : candidateMoves) {
            auto node = nodes[static_cast<size_t>(get<1>(move))];
            auto neigh = nodes[static_cast<size_t>(get<2>(move))];
            auto neighParent = neigh->parent;
            // check due to earlier moves
            if (TreeNode::IsAncestor(node, neighParent)) continue;
            DTYPE pathLengthDelta =
                pathLengths[neighParent->id] + Dist(node->loc, neighParent->loc) - pathLengths[node->id];
            if (pathLengthDelta > slacks[node->id]) continue;
            auto steinerPt = GetNearestPoint(node, neigh);
            DTYPE wireLengthDelta = Dist(node->loc, steinerPt) - node->WireToParent();
            if (wireLengthDelta >= 0) continue;
            // break
            Disconnect(node);
            // reroot
            if (steinerPt == neigh->loc) {
                Connect(node, neigh);
            } else if (steinerPt == neighParent->loc) {
                Connect(node, neighParent);
            } else {
                auto steinerNode = make_shared<TreeNode>(steinerPt);
                Connect(steinerNode, neighParent);
                Disconnect(neigh);
                Connect(neigh, steinerNode);
                Connect(node, steinerNode);
                // for later moves
                steinerNode->id = nodes.size();
                nodes.push_back(steinerNode);
                pathLengths.push_back(pathLengths[neighParent->id] + steinerNode->WireToParent());
                slacks.push_back(Dist(steinerNode->loc, tree.source->loc) * (1 + eps) - pathLengths.back());
            }
            // update slack for later moves: first subtree, then path to source
            updatePathLengthsDfs(updatePathLengthsDfs, neighParent);
            updateSlacksDfs(updateSlacksDfs, neighParent);
            auto tmp = neighParent;
            while (tmp->parent) {
                slacks[tmp->parent->id] = min(slacks[tmp->parent->id], slacks[tmp->id]);
                tmp = tmp->parent;
            }
        }

        // Finalize
        // tree.RemoveTopoRedundantSteiner();
        vector<shared_ptr<TreeNode>> cleanupNodes;
        cleanupNodes.reserve(nodes.size());
        auto collectCleanupNodes = [&](auto&& self, const shared_ptr<TreeNode>& node) -> void {
            for (const auto& child : node->children) self(self, child);
            cleanupNodes.push_back(node);
        };
        collectCleanupNodes(collectCleanupNodes, tree.source);
        for (const auto& node : cleanupNodes) {
            // degree may change after post-order traversal of its children
            if (node->pin) continue;
            if (node->children.empty()) {
                Disconnect(node);
            } else if (node->children.size() == 1) {
                auto oldParent = node->parent, oldChild = node->children[0];
                Disconnect(node);
                Disconnect(oldChild);
                Connect(oldChild, oldParent);
            }
        }
        ++substituteRounds;
    }
}

}  // namespace salt
