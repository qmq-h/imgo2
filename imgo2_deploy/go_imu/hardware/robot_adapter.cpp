#include "robot_adapter.h"
#include <cmath>
#include <set>
#include <stdexcept>
namespace imgo2_sdk {
RobotAdapter::RobotAdapter(std::vector<int> mapping):map_(std::move(mapping)) {
    std::set<int> ids(map_.begin(),map_.end());
    if(map_.size()!=12 || ids.size()!=12 || *ids.begin()!=0 || *ids.rbegin()!=11)
        throw std::invalid_argument("invalid physical joint mapping");
}
std::vector<float> RobotAdapter::positionsToHardware(const std::vector<float>& values) const {
    if(values.size()!=12) throw std::invalid_argument("mapping requires 12 values");
    std::vector<float> out(12);
    for(size_t i=0;i<12;++i) {
        if(!std::isfinite(values[i])) throw std::invalid_argument("nonfinite mapped value");
        out[map_[i]]=values[i];
    }
    return out;
}
go::Targets RobotAdapter::toHardware(const go::Targets& p) const {
    go::Targets h;h.q=positionsToHardware(p.q);h.dq=positionsToHardware(p.dq);h.tau=positionsToHardware(p.tau);
    h.kp=positionsToHardware(p.kp);h.kd=positionsToHardware(p.kd);return h;
}
RobotState RobotAdapter::toPolicy(const real_hardware::HardwareState& h) const {
    RobotState s;
    if(!h.started || h.faulted || !h.imu_valid || h.joints.size()!=12) return s;
    const auto& imu=h.imu.data;
    if(imu.quaternion.size()!=4 || imu.gyroscope.size()!=3) return s;
    s.q.resize(12);s.dq.resize(12);
    for(size_t i=0;i<12;++i) {
        const auto& j=h.joints[map_[i]];
        if(!j.valid || !std::isfinite(j.feedback.q) || !std::isfinite(j.feedback.dq)) return RobotState{};
        s.q[i]=j.feedback.q;s.dq[i]=j.feedback.dq;s.joint_receive_ns[i]=j.receive_time_ns;
    }
    std::copy(imu.quaternion.begin(),imu.quaternion.end(),s.quaternion.begin());
    std::copy(imu.gyroscope.begin(),imu.gyroscope.end(),s.gyro.begin());
    for(float x:s.quaternion) if(!std::isfinite(x)) return RobotState{};
    for(float x:s.gyro) if(!std::isfinite(x)) return RobotState{};
    s.imu_receive_ns=imu.receive_time_ns;s.captured_at=Clock::now();
    s.valid=true;return s; // No direction/offset/ratio or IMU rotation here.
}
}
