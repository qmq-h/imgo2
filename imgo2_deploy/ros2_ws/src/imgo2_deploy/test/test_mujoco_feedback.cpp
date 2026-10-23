#include "mujoco_feedback.hpp"
#include <iostream>
#include <numeric>
#include <limits>
void check(bool x,const char* why){if(!x)throw std::runtime_error(why);}
int main(int argc,char** argv)
{
 try {
  char error[2048]{};auto*m=mj_loadXML(argv[1],nullptr,error,sizeof(error));check(m,error);auto*d=mj_makeData(m);
  std::vector<int> map(12);std::iota(map.begin(),map.end(),0);
  int qi=mj_name2id(m,mjOBJ_SENSOR,"orientation"),gi=mj_name2id(m,mjOBJ_SENSOR,"angular_velocity");
  m->sensor_noise[qi]=0;m->sensor_noise[gi]=0;
  auto source=[&](double t){d->time=t;for(int i=0;i<12;i++){d->sensordata[i]=t;d->sensordata[12+i]=t+.1;d->sensordata[24+i]=t+.2;}d->sensordata[36]=1;for(int i=37;i<40;i++)d->sensordata[i]=0;for(int i=40;i<43;i++)d->sensordata[i]=t;};
  MujocoFeedback fb;MujocoFeedback::State out;
  fb.Configure(m,map,1.5,3.3);source(0);fb.Sample(m,d);check(!fb.Read(0,out),"delay warmup must not leak current data");
  for(int k=1;k<=20;k++){source(k*.0005);fb.Sample(m,d);}
  check(fb.Read(.01,out),"delayed frame ready");
  check(std::abs(out.motor_time-.0085)<1e-12 && std::abs(out.motors.q[0]-.0085)<1e-12,"fractional motor delay");
  check(std::abs(out.imu_time-.005)<1e-12 && std::abs(out.imu.gyro[0]-.005)<1e-12,"IMU delay/sample hold");
  check(std::abs(out.motors.dq[0]-(out.motors.q[0]+.1))<1e-12 && std::abs(out.motors.tau[0]-(out.motors.q[0]+.2))<1e-12,"motor q/dq/tau coherent");
  source(0);fb.Sample(m,d);check(!fb.Read(0,out),"reset must clear old history");
  fb.Configure(m,map,10,0);
  for(int k=0;k<=100;++k){source(k*.0005);fb.Sample(m,d);}
  check(fb.Read(.05,out) && std::abs(out.motor_time-.04)<1e-12 && std::abs(out.motors.q[0]-.04)<1e-12,
        "10ms configured feedback must deliver the state from 40ms at time 50ms");
  bool rejected=false;try{fb.Configure(m,map,std::numeric_limits<double>::infinity(),0);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"invalid delay rejection");
  fb.Configure(m,map,0,0);source(0);fb.Sample(m,d);check(fb.Read(0,out),"zero delay immediate");
  source(.001);fb.Sample(m,d);check(fb.Read(.001,out)&&out.imu_time==0,"IMU source sample rate");
  m->sensor_noise[qi]=.02;m->sensor_noise[gi]=.005;
  fb.Configure(m,map,0,0);double sum=0,sumsq=0;std::array<double,4> first{};
  for(int k=0;k<4000;k++){
   source(k*.005);for(int i=40;i<43;i++)d->sensordata[i]=0;fb.Sample(m,d);check(fb.Read(d->time,out),"noise frame ready");
   if(k==0)first=out.imu.quat;
   double norm=0;for(double v:out.imu.quat)norm+=v*v;check(std::abs(norm-1)<1e-12,"noisy quaternion normalization");
   for(double v:out.imu.gyro){sum+=v;sumsq+=v*v;}
   auto frozen=out;fb.Sample(m,d);fb.Read(d->time,out);check(out.imu.quat==frozen.imu.quat&&out.imu.gyro==frozen.imu.gyro,"paused time must not regenerate noise");
  }
  const double rms=std::sqrt(sumsq/12000);check(std::abs(sum/12000)<.0003&&rms>.0047&&rms<.0053,"noise statistics");
  fb.Reset();source(0);fb.Sample(m,d);fb.Read(0,out);check(first==out.imu.quat,"reset noise seed reproducibility");
  mj_deleteData(d);mj_deleteModel(m);
  std::cout<<"fractional delays / causal warmup / reset / sample rate / frozen pause / quaternion norm / noise statistics PASS; gyro rms="<<rms<<'\n';return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
