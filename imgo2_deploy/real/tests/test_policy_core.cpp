#include "sdk/rl_sdk.h"
#include "hardware/robot_adapter.h"
#include "apps/deployment_input.h"
#include "function/my_myth.h"
#include <cmath>
#include <iostream>
#include <limits>
#define CHECK(x) do{if(!(x)){std::cerr<<#x<<'\n';return 1;}}while(0)
using namespace imgo2_sdk;
int main(int argc,char** argv) {
    CHECK(argc==2);auto c=PolicyConfig::load(argv[1]);
    CHECK(c.fixed_kp[0]==60 && c.fixed_kd[0]==2 && c.kp[0]==25 && c.kd[0]>0);
    RobotState s;s.valid=true;s.q=c.default_pose;s.dq.assign(12,2);s.gyro={{1,2,3}};
    auto obs=buildObservation(s,{{.2f,-.1f,.3f}},std::vector<float>(12,.4f),c);
    CHECK(obs.size()==45 && obs[0]==.25f && obs[2]==.75f);
    CHECK(obs[3]==0 && obs[4]==0 && obs[5]==-1 && obs[6]==.2f && obs[9]==0);
    CHECK(obs[21]==.1f && obs[33]==.4f);
    auto clipped=clipActions(std::vector<float>(12,10),c);auto target=convertAction(clipped,c);
    CHECK(clipped[0]==3 && target.q[0]==.375f && std::abs(target.q[1]-1.62f)<1e-6);
    CHECK(target.kp[0]==c.kp[0] && target.kd[0]==c.kd[0] && target.tau[0]==0);
    bool rejected=false;try{clipActions(std::vector<float>(12,std::numeric_limits<float>::quiet_NaN()),c);}catch(const std::exception&){rejected=true;}CHECK(rejected);
    const float root=std::sqrt(.5f);auto g=my_myth::projectedGravity({{root,0,root,0}});
    CHECK(std::abs(g[0]-1)<1e-6 && std::abs(g[2])<1e-6);
    std::vector<int> map={3,4,5,0,1,2,9,10,11,6,7,8};RobotAdapter adapter(map);
    auto raw=real_hardware::HardwareState{};raw.started=true;raw.imu_valid=true;raw.joints.resize(12);
    for(size_t i=0;i<12;++i){raw.joints[i].valid=true;raw.joints[i].feedback.q=float(i);raw.joints[i].feedback.dq=-float(i);}
    auto mapped=adapter.toPolicy(raw);CHECK(mapped.valid && mapped.q[0]==3 && mapped.dq[0]==-3);
    CHECK(adapter.positionsToHardware(mapped.q)==std::vector<float>({0,1,2,3,4,5,6,7,8,9,10,11}));
    raw.joints[1].valid=false;CHECK(!adapter.toPolicy(raw).valid);
    go_app::DeploymentInput keys;
    CHECK(keys.feed('w').kind==go_app::KeyEvent::Kind::Velocity);
    CHECK(keys.feed('0').kind==go_app::KeyEvent::Kind::None);
    CHECK(keys.feed('\n').digit==0);
    keys.feed('1');keys.feed('0');CHECK(keys.feed('\n').kind==go_app::KeyEvent::Kind::None);
    keys.feed('1');keys.feed('\b');keys.feed('9');CHECK(keys.feed('\n').digit==9);
    keys.feed('0');CHECK(keys.feed('A').key=='a');CHECK(keys.feed('\n').kind==go_app::KeyEvent::Kind::None);
    CHECK(keys.feed(' ').kind==go_app::KeyEvent::Kind::Velocity);
    for(char key:std::string("\033[A\033[D\033[1;5A")) CHECK(keys.feed(key).kind==go_app::KeyEvent::Kind::None);
    CHECK(keys.feed('w').kind==go_app::KeyEvent::Kind::Velocity);
    std::cout<<"policy math/mapping/input passed\n";
}
