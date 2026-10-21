#pragma once
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

// ReadYaml merges keys: prevent the AMP-only key leaking into another policy.
inline bool UseAmpJointTargetClip(const std::string& active_config, bool enabled)
{
    return active_config == "amp" && enabled;
}

inline std::vector<float> ClipPolicyJointTargets(
    const std::vector<float>& targets, const std::vector<float>& lower,
    const std::vector<float>& upper)
{
    if (targets.size() != lower.size() || targets.size() != upper.size())
        throw std::invalid_argument("Joint target clip dimensions do not match");
    std::vector<float> result(targets);
    for (std::size_t i=0; i<targets.size(); ++i)
    {
        if (lower[i] > upper[i]) throw std::invalid_argument("Invalid joint target clip bounds");
        result[i] = std::clamp(targets[i], lower[i], upper[i]);
    }
    return result;
}
