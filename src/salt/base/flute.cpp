#include "flute.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>

#include "flute/flute.h" // should be included after boost/functional/hash.hpp

namespace {

struct PairHash {
    std::size_t operator()(const std::pair<DTYPE, DTYPE>& key) const {
        const std::uint64_t ux = static_cast<std::uint32_t>(key.first);
        const std::uint64_t uy = static_cast<std::uint32_t>(key.second);
        return static_cast<std::size_t>((ux << 32) ^ uy);
    }
};

int FluteAccuracy() {
    static const int accuracy = [] {
        const char* value = std::getenv("SALT_FLUTE_ACCURACY");
        if (!value || !*value) return ACCURACY;
        const int parsed = std::atoi(value);
        return parsed > 0 ? parsed : ACCURACY;
    }();
    return accuracy;
}

}  // namespace

void salt::FluteBuilder::Run(const salt::Net& net, salt::Tree& saltTree) {
    // load LUT
    static bool once = false;
    if (!once) {
        flute::readLUT();
        once = true;
    }

    // Obtain flute tree
    flute::Tree fluteTree;
    fluteTree.branch = nullptr;
    int d = net.pins.size();
    assert(d <= MAXD);
    int x[MAXD], y[MAXD];
    for (size_t i = 0; i < d; ++i) {
        x[i] = net.pins[i]->loc.x;
        y[i] = net.pins[i]->loc.y;
    }
    if (fluteTree.branch) free(fluteTree.branch);  // is it complete for mem leak?
    fluteTree = flute::flute(d, x, y, FluteAccuracy());

    // Build adjacency list
    unordered_map<pair<DTYPE, DTYPE>, shared_ptr<salt::TreeNode>, PairHash> key2node;
    for (auto p : net.pins) {
        key2node[{p->loc.x, p->loc.y}] = make_shared<salt::TreeNode>(p);
    }
    auto& t = fluteTree;

    auto FindOrCreate = [&](DTYPE x, DTYPE y) {
        auto it = key2node.find({x, y});
        if (it == key2node.end()) {
            shared_ptr<salt::TreeNode> node = make_shared<salt::TreeNode>(x, y);
            key2node[{x, y}] = node;
            return node;
        } else
            return it->second;
    };

    for (int i = 0; i < 2 * t.deg - 2; i++) {
        int j = t.branch[i].n;
        if (t.branch[i].x == t.branch[j].x && t.branch[i].y == t.branch[j].y) continue;
        // any more duplicate?
        shared_ptr<salt::TreeNode> n1 = FindOrCreate(t.branch[i].x, t.branch[i].y);
        shared_ptr<salt::TreeNode> n2 = FindOrCreate(t.branch[j].x, t.branch[j].y);
        // printlog(LOG_INFO, "%d - %d\n", n1->pin?n1->pin->id:-1, n2->pin?n2->pin->id:-1);
        n1->children.push_back(n2);
        n2->children.push_back(n1);
    }

    // Reverse parent-child orders
    saltTree.source = key2node[{net.source()->loc.x, net.source()->loc.y}];
    saltTree.SetParentFromUndirectedAdjList();
    saltTree.net = &net;

    free(fluteTree.branch);
}
