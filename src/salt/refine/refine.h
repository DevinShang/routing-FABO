#pragma once

#include "salt/base/tree.h"

namespace salt {

class Refine {
public:
    static void CancelIntersect(Tree& tree);
    static void Flip(Tree& tree);
    static void UShift(Tree& tree);  // should be after Flip to achieve good quality
    static void Substitute(Tree& tree, double eps, bool useRTree = true, int maxRounds = 0,
                           int linearCandidateCap = 0, int linearQueryMode = 0);
    static void SiblingPairSubstitute(Tree& tree, double eps, bool useRTree = true, bool allowNeutral = false);
    static void TripletBundleSubstitute(Tree& tree, double eps, bool allowNeutral = false);
    static void QuadBundleSubstitute(Tree& tree, double eps, bool allowNeutral = false);
};

}  // namespace salt
