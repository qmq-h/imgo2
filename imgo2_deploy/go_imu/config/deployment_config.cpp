#include "deployment_config.h"
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>
namespace imgo2_sdk {
DeploymentConfig::DeploymentConfig() {
    stand_pose={0,.87f,-1.82f,0,.87f,-1.82f,0,.87f,-1.82f,0,.87f,-1.82f};
    crouch_pose={HIP_START_OFFSET,THIGH_START_OFFSET,SHANK_START_OFFSET,
                -HIP_START_OFFSET,THIGH_START_OFFSET,SHANK_START_OFFSET,
                HIP_START_OFFSET,THIGH_START_OFFSET,SHANK_START_OFFSET,
                -HIP_START_OFFSET,THIGH_START_OFFSET,SHANK_START_OFFSET};
    policy_to_hardware.resize(12);std::iota(policy_to_hardware.begin(),policy_to_hardware.end(),0);
}
void DeploymentConfig::validate() const {
    if(control_ms<=0 || policy_ms<=0 || result_timeout_ms<policy_ms || torch_threads<1 ||
       !std::isfinite(stand_s) || stand_s<=0 || !std::isfinite(return_stand_s) || return_stand_s<=0 ||
       !std::isfinite(crouch_s) || crouch_s<=0 ||
       !std::isfinite(blend_s) || blend_s<=0 || !std::isfinite(settle_s) || settle_s<0 ||
       !std::isfinite(position_tolerance) || position_tolerance<=0 ||
       !std::isfinite(velocity_tolerance) || velocity_tolerance<=0)
        throw std::invalid_argument("invalid deployment times/tolerances");
    for(float v:velocity_limits) if(!std::isfinite(v) || v<=0) throw std::invalid_argument("invalid velocity limits");
    if(stand_pose.size()!=12) throw std::invalid_argument("stand requires 12 joints");
    for(float v:stand_pose) if(!std::isfinite(v)) throw std::invalid_argument("nonfinite stand pose");
    if(!std::isfinite(crouch_settle_s) || crouch_settle_s<0 ||
       !std::isfinite(crouch_arrival_timeout_s) || crouch_arrival_timeout_s<=crouch_settle_s ||
       !std::isfinite(crouch_position_tolerance) || crouch_position_tolerance<=0 ||
       !std::isfinite(crouch_velocity_tolerance) || crouch_velocity_tolerance<=0)
        throw std::invalid_argument("invalid crouch arrival limits/times");
    if(crouch_pose.size()!=12) throw std::invalid_argument("crouch requires 12 joints");
    for(float v:crouch_pose) if(!std::isfinite(v)) throw std::invalid_argument("nonfinite crouch pose");
    std::set<int> ids(policy_to_hardware.begin(),policy_to_hardware.end());
    if(policy_to_hardware.size()!=12 || ids.size()!=12 || *ids.begin()!=0 || *ids.rbegin()!=11)
        throw std::invalid_argument("policy_to_hardware must be a permutation of 0..11");
}
DeploymentConfig DeploymentConfig::load(const std::filesystem::path& path) {
    auto root=YAML::LoadFile(path.string())["deployment"];
    if(!root) throw std::invalid_argument("missing deployment section");
    DeploymentConfig c;
#define READ(field) if(root[#field]) c.field=root[#field].as<decltype(c.field)>();
    READ(control_ms) READ(policy_ms) READ(result_timeout_ms) READ(torch_threads)
    READ(stand_s) READ(return_stand_s) READ(crouch_s) READ(blend_s) READ(settle_s)
    READ(position_tolerance) READ(velocity_tolerance) READ(mapping_confirmed)
    READ(stand_pose) READ(crouch_pose) READ(policy_to_hardware)
    READ(crouch_settle_s) READ(crouch_arrival_timeout_s)
    READ(crouch_position_tolerance) READ(crouch_velocity_tolerance)
#undef READ
    if(root["velocity_limits"]) {
        auto v=root["velocity_limits"].as<std::vector<float>>();
        if(v.size()!=3) throw std::invalid_argument("velocity_limits requires xyz commands");
        std::copy(v.begin(),v.end(),c.velocity_limits.begin());
    }
    c.validate();return c;
}
}
