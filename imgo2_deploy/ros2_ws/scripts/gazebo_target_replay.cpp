// Diagnostic-only ModelPlugin: restore a snapshot, then publish recorded targets
// to the UNMODIFIED ROS2 joint controller. Never loads a policy or real hardware.
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/node.hpp>
#include <robot_msgs/msg/robot_command.hpp>
#include <robot_msgs/msg/robot_state.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <array>
#include <vector>

namespace gazebo {
class TargetReplay : public ModelPlugin {
  physics::ModelPtr model;
  physics::LinkPtr base;
  physics::JointPtr world_anchor;
  bool fixed_base=false;
  std::vector<physics::JointPtr> joints;
  gazebo_ros::Node::SharedPtr node;
  rclcpp::Publisher<robot_msgs::msg::RobotCommand>::SharedPtr pub;
  rclcpp::Subscription<robot_msgs::msg::RobotState>::SharedPtr sub;
  event::ConnectionPtr connection;
  struct Frame { double t; robot_msgs::msg::RobotCommand msg; };
  std::vector<Frame> frames;
  std::array<double,12> q{}, dq{};
  ignition::math::Pose3d pose;
  ignition::math::Vector3d linear, angular;
  std::ofstream log, sent;
  std::string directory;
  bool started=false, done=false;
  double start=0, last_log=-1, last_poll=-1;
  size_t index=0;
  std::mutex mutex;
  robot_msgs::msg::RobotState feedback;
  double feedback_wall=0;
  static std::vector<double> numbers(const std::string &line) {
    std::vector<double> v; std::stringstream s(line); std::string token;
    while(std::getline(s,token,',')) v.push_back(std::stod(token));
    return v;
  }
  void restore() {
    std::map<std::string,double> values;
    for(size_t i=0;i<12;++i) values[joints[i]->GetName()]=q[i];
    model->SetJointPositions(values);
    model->SetWorldPose(pose);
    model->SetLinearVel(linear); model->SetAngularVel(angular);
    for(size_t i=0;i<12;++i) joints[i]->SetVelocity(0,dq[i]);
  }
 public:
  void Load(physics::ModelPtr m, sdf::ElementPtr sdf) override {
    model=m; base=m->GetLink("base"); directory=sdf->Get<std::string>("directory");
    fixed_base=sdf->HasElement("fixed_base") && sdf->Get<bool>("fixed_base");
    if(!base) throw std::runtime_error("replay: base missing");
    for(auto leg:{"FL","FR","RL","RR"}) for(auto type:{"hip","thigh","shank"}) {
      auto joint=m->GetJoint(std::string(leg)+"_"+type+"_joint");
      if(!joint) throw std::runtime_error("replay: joint missing");
      joints.push_back(joint);
    }
    std::ifstream initial(directory+"/initial.csv"); std::string line; std::getline(initial,line);
    auto v=numbers(line); if(v.size()!=37) throw std::runtime_error("replay: initial requires 37 values");
    pose.Set(ignition::math::Vector3d(v[0],v[1],v[2]), ignition::math::Quaterniond(v[3],v[4],v[5],v[6]));
    linear.Set(v[7],v[8],v[9]); angular.Set(v[10],v[11],v[12]);
    for(size_t i=0;i<12;++i){q[i]=v[13+i]; dq[i]=v[25+i];}
    std::ifstream input(directory+"/targets.csv"); std::getline(input,line);
    while(std::getline(input,line)) {
      auto a=numbers(line); if(a.size()!=61) throw std::runtime_error("replay: invalid target row");
      Frame f; f.t=a[0]; f.msg.motor_command.resize(12);
      for(size_t i=0;i<12;++i) { auto &j=f.msg.motor_command[i];
        j.q=a[1+i]; j.dq=a[13+i]; j.tau=a[25+i];j.kp=a[37+i];j.kd=a[49+i]; }
      frames.push_back(f);
    }
    if(frames.size()<2 || frames.front().t>1e-8) throw std::runtime_error("replay: targets must start at zero");
    node=gazebo_ros::Node::Get(sdf);
    pub=node->create_publisher<robot_msgs::msg::RobotCommand>("/robot_joint_controller/command",10);
    sub=node->create_subscription<robot_msgs::msg::RobotState>("/robot_joint_controller/state",rclcpp::SensorDataQoS(),
      [this](robot_msgs::msg::RobotState::SharedPtr msg) {std::lock_guard<std::mutex> lock(mutex);feedback=*msg;});
    log.open(directory+"/replay.csv"); log<<std::setprecision(17);
    log<<"t_s,sim_s,target_index,x,y,z,quat_w,quat_x,quat_y,quat_z,gyro_x,gyro_y,gyro_z,vx,vy,vz";
    for(auto key:{"q","dq","tau","controller_q","controller_dq","controller_tau"}) for(int i=0;i<12;++i)log<<','<<key<<'_'<<std::setw(2)<<std::setfill('0')<<i;
    log<<'\n'; sent.open(directory+"/published.csv"); sent<<std::setprecision(17)<<"t_s,index,scheduled_s\n";
    restore();
    connection=event::Events::ConnectWorldUpdateBegin(std::bind(&TargetReplay::update,this));
    std::ofstream(directory+"/plugin_ready")<<"ready\n";
  }
  void update() {
    const double now=model->GetWorld()->SimTime().Double();
    if(!started) {
      restore();
      if(now-last_poll>.05){last_poll=now;if(std::ifstream(directory+"/release")) {
        started=true;start=now;restore();
        if(fixed_base) {
          world_anchor=model->GetWorld()->Physics()->CreateJoint("fixed",model);
          world_anchor->SetName("diagnostic_world_base_anchor");
          world_anchor->Load(physics::LinkPtr(),base,ignition::math::Pose3d::Zero);
          world_anchor->Init();
        }
        pub->publish(frames[0].msg);sent<<0<<",0,0\n";
      }}
      return;
    }
    const double t=now-start;
    while(index+1<frames.size() && frames[index+1].t<=t+1e-9) {
      ++index;pub->publish(frames[index].msg);sent<<t<<','<<index<<','<<frames[index].t<<'\n';
    }
    if(now-last_log>=.000999) {
      last_log=now; const auto p=base->WorldPose(); const auto g=p.Rot().RotateVectorReverse(base->WorldAngularVel());
      const auto v=base->WorldLinearVel();
      log<<t<<','<<now<<','<<index<<','<<p.Pos().X()<<','<<p.Pos().Y()<<','<<p.Pos().Z()<<','<<p.Rot().W()<<','<<p.Rot().X()<<','<<p.Rot().Y()<<','<<p.Rot().Z()<<','<<g.X()<<','<<g.Y()<<','<<g.Z()<<','<<v.X()<<','<<v.Y()<<','<<v.Z();
      for(auto j:joints)log<<','<<j->Position(0);
      for(auto j:joints)log<<','<<j->GetVelocity(0);
      for(auto j:joints)log<<','<<j->GetForce(0);
      std::lock_guard<std::mutex> lock(mutex);
      for(int k=0;k<3;++k)for(int i=0;i<12;++i) {
        if(feedback.motor_state.size()!=12)log<<",nan";
        else {auto &j=feedback.motor_state[i];log<<','<<(k==0?j.q:k==1?j.dq:j.tau_est);}
      }
      log<<'\n';
    }
    if(!done && t>=frames.back().t+.02) {
      done=true;log.flush();sent.flush();std::ofstream(directory+"/done")<<t<<'\n';
    }
  }
};
GZ_REGISTER_MODEL_PLUGIN(TargetReplay)
}
