#include "policy_config.h"
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <stdexcept>
namespace imgo2_sdk {
namespace {
std::vector<float> values(const YAML::Node& root,const char* key,size_t count) {
    if(!root[key] || !root[key].IsSequence()) throw std::invalid_argument(std::string("missing policy vector: ")+key);
    auto out=root[key].as<std::vector<float>>();
    if(out.size()!=count) throw std::invalid_argument(std::string("invalid policy vector length: ")+key);
    for(float v:out) if(!std::isfinite(v)) throw std::invalid_argument(std::string("nonfinite policy value: ")+key);
    return out;
}
float scalar(const YAML::Node& root,const char* key) {
    float v=root[key].as<float>();
    if(!std::isfinite(v) || v<=0) throw std::invalid_argument(std::string("invalid policy scale: ")+key);
    return v;
}
}
PolicyConfig PolicyConfig::load(const std::filesystem::path& dir) {
    const auto root=YAML::LoadFile((dir/"config.yaml").string())["imgo2/ppo"];
    if(!root) throw std::invalid_argument("missing imgo2/ppo policy section");
    if(root["num_of_dofs"].as<int>()!=12 || root["num_observations"].as<int>()!=45)
        throw std::invalid_argument("only 45 observation / 12 position-action PPO is supported");
    const std::vector<std::string> expected={"ang_vel","gravity_vec","commands","dof_pos","dof_vel","actions"};
    if(root["observations"].as<std::vector<std::string>>()!=expected)
        throw std::invalid_argument("unsupported PPO observation order");
    if(!root["observations_history"] || !root["observations_history"].IsSequence() || root["observations_history"].size()!=0 ||
       !root["wheel_indices"] || !root["wheel_indices"].IsSequence() || root["wheel_indices"].size()!=0)
        throw std::invalid_argument("history and wheel policies are not supported");
    PolicyConfig c;
    c.default_pose=values(root,"default_dof_pos",12);c.action_scale=values(root,"action_scale",12);
    c.lower=values(root,"clip_actions_lower",12);c.upper=values(root,"clip_actions_upper",12);
    c.kp=values(root,"rl_kp",12);c.kd=values(root,"rl_kd",12);
    c.fixed_kp=values(root,"fixed_kp",12);c.fixed_kd=values(root,"fixed_kd",12);
    c.torque_limits=values(root,"torque_limits",12);
    auto scale=values(root,"commands_scale",3);std::copy(scale.begin(),scale.end(),c.commands_scale.begin());
    c.ang_vel_scale=scalar(root,"ang_vel_scale");c.dof_pos_scale=scalar(root,"dof_pos_scale");
    c.dof_vel_scale=scalar(root,"dof_vel_scale");c.clip_obs=scalar(root,"clip_obs");
    for(size_t i=0;i<12;++i) {
        if(c.lower[i]>c.upper[i] || c.action_scale[i]<=0 || c.kp[i]<0 || c.kd[i]<0 ||
           c.fixed_kp[i]<0 || c.fixed_kd[i]<0 || c.torque_limits[i]<=0)
            throw std::invalid_argument("invalid policy action bounds/gains");
    }
    for(float s:c.commands_scale) if(s<=0) throw std::invalid_argument("invalid commands_scale");
    const auto model=root["model_name"].as<std::string>();
    if(model.empty()) throw std::invalid_argument("empty model_name");
    c.model_path=std::filesystem::absolute(dir/model);
    if(!std::filesystem::is_regular_file(c.model_path)) throw std::invalid_argument("model not found: "+c.model_path.string());
    return c;
}
}
