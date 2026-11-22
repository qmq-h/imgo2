// Diagnostic only: candidate passive joint resistance added to original PD.
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
namespace gazebo {
class RackResistance : public ModelPlugin {
  struct Axis { physics::JointPtr joint; double fs,b,va; };
  std::vector<Axis> axes;
  event::ConnectionPtr connection;
  physics::ModelPtr model;
  std::ofstream log;
  double last=-1;
 public:
  void Load(physics::ModelPtr m,sdf::ElementPtr sdf) override {
    model=m;
    std::ifstream f(sdf->Get<std::string>("parameters"));std::string line;std::getline(f,line);
    while(std::getline(f,line)) {
      std::stringstream ss(line);std::string name,a,b,v;
      std::getline(ss,name,',');std::getline(ss,a,',');std::getline(ss,b,',');std::getline(ss,v,',');
      auto joint=m->GetJoint(name);double fs=std::stod(a),visc=std::stod(b),va=std::stod(v);
      if(!joint || fs<0 || visc<0 || va<=0) throw std::runtime_error("Invalid rack resistance parameter");
      axes.push_back({joint,fs,visc,va});
    }
    if(axes.empty())throw std::runtime_error("No rack resistance axes");
    log.open(sdf->Get<std::string>("log"));log<<std::setprecision(17)<<"sim_s";
    for(auto &a:axes)for(auto key:{"q","dq","resistance"})log<<','<<a.joint->GetName()<<'_'<<key;
    log<<'\n';
    connection=event::Events::ConnectWorldUpdateBegin(std::bind(&RackResistance::update,this));
  }
  void update() {
    const double now=model->GetWorld()->SimTime().Double();const bool write_log=now-last>=.004999;
    if(write_log){last=now;log<<now;}
    for(auto &a:axes){
      const double v=a.joint->GetVelocity(0);const double resistance=a.fs*std::tanh(v/a.va)+a.b*v;
      // Gazebo Joint::SetForce is additive within a physics step.
      a.joint->SetForce(0,-resistance);
      if(write_log)log<<','<<a.joint->Position(0)<<','<<v<<','<<resistance;
    }
    if(write_log){log<<'\n';log.flush();}
  }
};
GZ_REGISTER_MODEL_PLUGIN(RackResistance)
}
