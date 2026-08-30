#include "salt.h"

#include "base/flute.h"
#include "base/rsa.h"
#include "refine/refine.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace salt {
namespace {

struct SaltProfileRun {
    int net_id = -1;
    int degree = 0;
    int refine_level = 0;
    std::chrono::high_resolution_clock::time_point start;
    std::map<std::string, long long> phase_us;
    std::map<std::string, int> phase_count;
};

SaltProfileRun*& SaltProfileCurrent() {
    static SaltProfileRun* current = nullptr;
    return current;
}

bool SaltProfileEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("SALT_PROFILE_PHASES");
        return value && std::atoi(value) != 0;
    }();
    return enabled;
}

std::ostream& SaltProfileStream() {
    static std::ofstream ofs;
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        const char* path = std::getenv("SALT_PROFILE_OUT");
        if (path && *path) ofs.open(path, std::ios::out | std::ios::app);
    }
    return ofs.is_open() ? ofs : std::cerr;
}

void SaltProfileAdd(const char* label, long long us) {
    SaltProfileRun* current = SaltProfileCurrent();
    if (!current) return;
    current->phase_us[label] += us;
    current->phase_count[label] += 1;
}

struct SaltProfileScope {
    explicit SaltProfileScope(const char* label)
        : active(SaltProfileEnabled() && SaltProfileCurrent()),
          phase(label),
          start(active ? std::chrono::high_resolution_clock::now() :
                         std::chrono::high_resolution_clock::time_point{}) {}

    ~SaltProfileScope() {
        if (!active) return;
        const auto end = std::chrono::high_resolution_clock::now();
        const long long us =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        SaltProfileAdd(phase.c_str(), us);
    }

    bool active = false;
    std::string phase;
    std::chrono::high_resolution_clock::time_point start;
};

void SaltProfileEmit(const SaltProfileRun& run) {
    std::ostream& os = SaltProfileStream();
    os << "{\"event\":\"salt_phase_profile\""
       << ",\"net_id\":" << run.net_id
       << ",\"degree\":" << run.degree
       << ",\"refine_level\":" << run.refine_level
       << ",\"phase_us\":{";
    bool first = true;
    for (const auto& kv : run.phase_us) {
        if (!first) os << ',';
        first = false;
        os << "\"" << kv.first << "\":" << kv.second;
    }
    os << "},\"phase_count\":{";
    first = true;
    for (const auto& kv : run.phase_count) {
        if (!first) os << ',';
        first = false;
        os << "\"" << kv.first << "\":" << kv.second;
    }
    os << "}}\n";
}

struct SaltProfileRunScope {
    SaltProfileRun run;
    SaltProfileRun* previous = nullptr;
    bool active = false;

    SaltProfileRunScope(const Net& net, int refine_level) : active(SaltProfileEnabled()) {
        if (!active) return;
        run.net_id = net.id;
        run.degree = static_cast<int>(net.pins.size());
        run.refine_level = refine_level;
        run.start = std::chrono::high_resolution_clock::now();
        previous = SaltProfileCurrent();
        SaltProfileCurrent() = &run;
    }

    ~SaltProfileRunScope() {
        if (!active) return;
        const auto end = std::chrono::high_resolution_clock::now();
        run.phase_us["run.total"] +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - run.start).count();
        run.phase_count["run.total"] += 1;
        SaltProfileEmit(run);
        SaltProfileCurrent() = previous;
    }
};

}  // namespace

void SaltBase::Init(Tree& minTree, shared_ptr<Pin> srcP) {
    SaltProfileScope profile("init");
    minTree.UpdateId();
    auto mtNodes = minTree.ObtainNodes();
    slNodes.resize(mtNodes.size());
    shortestDists.resize(mtNodes.size());
    curDists.resize(mtNodes.size());
    for (auto mtN : mtNodes) {
        slNodes[mtN->id] = make_shared<TreeNode>(mtN->loc, mtN->pin, mtN->id);
        shortestDists[mtN->id] = Dist(mtN->loc, srcP->loc);
        curDists[mtN->id] = numeric_limits<DTYPE>::max();
    }
    curDists[srcP->id] = 0;
    slSrc = slNodes[srcP->id];
}

void SaltBase::Finalize(const Net& net, Tree& tree) {
    SaltProfileScope profile("finalize");
    for (auto n : slNodes)
        if (n->parent) slNodes[n->parent->id]->children.push_back(n);
    tree.source = slSrc;
    tree.net = &net;
}

void SaltBuilder::Run(const Net& net, Tree& tree, double eps, int refineLevel) {
    SaltProfileRunScope profile_run(net, refineLevel);
    // SMT
    Tree smt;
    FluteBuilder fluteB;
    {
        SaltProfileScope profile("smt.flute_run");
        fluteB.Run(net, smt);
    }

    // Refine SMT
    if (refineLevel >= 1) {
        {
            SaltProfileScope profile("smt.refine_flip");
            Refine::Flip(smt);
        }
        {
            SaltProfileScope profile("smt.refine_ushift");
            Refine::UShift(smt);
        }
    }

    // Init
    Init(smt, net.source());

    // DFS
    {
        SaltProfileScope profile("salt.dfs");
        DFS(smt.source, slSrc, eps);
    }
    Finalize(net, tree);
    {
        SaltProfileScope profile("salt.remove_topo_redundant");
        tree.RemoveTopoRedundantSteiner();
    }

    // Connect breakpoints to source by RSA
    salt::RsaBuilder rsaB;
    {
        SaltProfileScope profile("salt.replace_root_children_rsa");
        rsaB.ReplaceRootChildren(tree);
    }

    // Refine SALT
    if (refineLevel >= 1) {
        {
            SaltProfileScope profile("salt.refine_cancel_intersect");
            Refine::CancelIntersect(tree);
        }
        {
            SaltProfileScope profile("salt.refine_flip");
            Refine::Flip(tree);
        }
        {
            SaltProfileScope profile("salt.refine_ushift");
            Refine::UShift(tree);
        }
        if (refineLevel >= 2) {
            SaltProfileScope profile("salt.refine_substitute");
            Refine::Substitute(tree, eps, refineLevel == 3);
        }
    }
}

bool SaltBuilder::Relax(const shared_ptr<TreeNode>& u, const shared_ptr<TreeNode>& v) {
    DTYPE newDist = curDists[u->id] + Dist(u->loc, v->loc);
    if (curDists[v->id] > newDist) {
        curDists[v->id] = newDist;
        v->parent = u;
        return true;
    } else if (curDists[v->id] == newDist && Dist(u->loc, v->loc) < v->WireToParentChecked()) {
        v->parent = u;
        return true;
    } else
        return false;
}

void SaltBuilder::DFS(const shared_ptr<TreeNode>& smtNode, const shared_ptr<TreeNode>& slNode, double eps) {
    if (smtNode->pin && curDists[slNode->id] > (1 + eps) * shortestDists[slNode->id]) {
        slNode->parent = slSrc;
        curDists[slNode->id] = shortestDists[slNode->id];
    }
    for (auto c : smtNode->children) {
        Relax(slNode, slNodes[c->id]);
        DFS(c, slNodes[c->id], eps);
        Relax(slNodes[c->id], slNode);
    }
}

}  // namespace salt
