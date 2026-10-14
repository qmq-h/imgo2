#include "../src/imgo2_deploy/include/policy_action_clip.hpp"
#include <cmath>
#include <iostream>

int main()
{
    if (!UseAmpJointTargetClip("amp",true) || UseAmpJointTargetClip("amp",false)
        || UseAmpJointTargetClip("ppo",true) || UseAmpJointTargetClip("himloco",true))
        throw std::runtime_error("AMP-only mode leaks across merged config keys");
    const std::vector<float> raw{-5, 5, 0, -20, 20};
    const std::vector<float> scale{.25f,.25f,.125f,.25f,.25f};
    const std::vector<float> offset{-1.82f,.87f,0,-1.82f,.87f};
    std::vector<float> targets;
    for (std::size_t i=0; i<raw.size(); ++i) targets.push_back(raw[i]*scale[i]+offset[i]);
    const auto clipped=ClipPolicyJointTargets(targets,std::vector<float>(5,-3),std::vector<float>(5,3));
    const std::vector<float> expected{-3,2.12f,0,-3,3};
    for (std::size_t i=0; i<raw.size(); ++i)
        if (std::fabs(clipped[i]-expected[i])>1e-6)
            throw std::runtime_error("Scale/offset/target clip mismatch");
    if (raw[0]!=-5 || raw[1]!=5) throw std::runtime_error("Raw feedback was modified");
    for (int invalid=0; invalid<2; ++invalid)
    {
        bool threw=false;
        try { ClipPolicyJointTargets({0},invalid ? std::vector<float>{1}:std::vector<float>{}, {0}); }
        catch (const std::invalid_argument&) { threw=true; }
        if (!threw) throw std::runtime_error("Invalid target bounds accepted");
    }
    std::cout << "AMP target clip, raw feedback and cross-policy isolation passed\n";
}
