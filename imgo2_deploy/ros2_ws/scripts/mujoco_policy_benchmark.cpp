// Headless deterministic-time policy comparison using the deployment PD/feedback classes.
// Does not run the GUI/FSM threads, train, export, or overwrite policy/configuration files.
#include <torch/script.h>
#include <ATen/Parallel.h>
#include <mujoco/mujoco.h>
#include <yaml-cpp/yaml.h>
#include "mujoco_motor_pd.hpp"
#include "mujoco_feedback.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static void BodyVector(const mjtNum* rotation, const double* world, double* body) {
    for (int i=0;i<3;++i) {body[i]=0; for(int j=0;j<3;++j)body[i]+=rotation[3*j+i]*world[j];}
}
static std::array<double,3> Rpy(const mjtNum* q) {
    const double w=q[0],x=q[1],y=q[2],z=q[3];
    return {std::atan2(2*(w*x+y*z),1-2*(x*x+y*y)),
            std::asin(std::clamp(2*(w*y-z*x),-1.,1.)),
            std::atan2(2*(w*z+x*y),1-2*(y*y+z*z))};
}
int main(int argc,char** argv) {
    if(argc!=10){std::cerr<<"scene policy base.yaml ppo.yaml csv vx vy wz seconds\n";return 2;}
    mjModel* m=nullptr; mjData* d=nullptr;
    try {
        torch::NoGradGuard guard; torch::set_num_threads(1);
        auto base=YAML::LoadFile(argv[3])["imgo2"];
        auto cfg=YAML::LoadFile(argv[4])["imgo2/ppo"];
        const auto mapping=cfg["joint_mapping"].as<std::vector<int>>();
        const auto q0=cfg["default_dof_pos"].as<std::vector<float>>();
        const auto kp=cfg["rl_kp"].as<std::vector<float>>(),kd=cfg["rl_kd"].as<std::vector<float>>();
        const auto fixed_kp=cfg["fixed_kp"].as<std::vector<float>>(),fixed_kd=cfg["fixed_kd"].as<std::vector<float>>();
        const auto scale=cfg["action_scale"].as<std::vector<float>>();
        const auto low=cfg["clip_actions_lower"].as<std::vector<float>>(),high=cfg["clip_actions_upper"].as<std::vector<float>>();
        const auto names=base["joint_names"].as<std::vector<std::string>>();
        const double control_dt=base["dt"].as<double>();
        const double policy_dt=control_dt*base["decimation"].as<int>();
        const double motor_delay=base["mujoco_motor_feedback_delay_ms"].as<double>();
        const double imu_delay=base["mujoco_imu_feedback_delay_ms"].as<double>();
        const float ang_scale=cfg["ang_vel_scale"].as<float>(),dq_scale=cfg["dof_vel_scale"].as<float>();
        const float q_scale=cfg["dof_pos_scale"].as<float>(),clip_obs=cfg["clip_obs"].as<float>();
        if(cfg["num_observations"].as<int>()!=45 || mapping.size()!=12 || q0.size()!=12 || cfg["observations_history"].size())
            throw std::runtime_error("Only the configured 45D feedforward PPO interface is supported");
        if(cfg["observations"].as<std::vector<std::string>>()!=std::vector<std::string>{"ang_vel","gravity_vec","commands","dof_pos","dof_vel","actions"})
            throw std::runtime_error("Unexpected observation order");
        const auto cmd_scale=cfg["commands_scale"].as<std::vector<float>>();
        const double command[3]={std::stod(argv[6]),std::stod(argv[7]),std::stod(argv[8])};
        const double seconds=std::stod(argv[9]),settle_s=2.,zero_policy_s=3.;
        auto policy=torch::jit::load(argv[2],torch::kCPU);policy.eval();
        char error[2048]={};m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
        if(!m)throw std::runtime_error(error);
        d=mj_makeData(m);
        const int base_id=mj_name2id(m,mjOBJ_BODY,"base");
        const int imu_id=mj_name2id(m,mjOBJ_SITE,"imu");
        const int floor=mj_name2id(m,mjOBJ_GEOM,"floor");
        if(base_id<0||imu_id<0||floor<0)throw std::runtime_error("Missing base, IMU site or floor");
        std::array<int,12> qadr{},vadr{},jids{};
        for(int i=0;i<12;++i){
            int joint=mj_name2id(m,mjOBJ_JOINT,names[i].c_str());
            if(joint<0||m->actuator_trnid[2*mapping[i]]!=joint)throw std::runtime_error("Joint/actuator mapping mismatch");
            jids[i]=joint;qadr[i]=m->jnt_qposadr[joint];vadr[i]=m->jnt_dofadr[joint];d->qpos[qadr[i]]=q0[i];
        }
        const std::array<std::string,4> legs={"FL","FR","RL","RR"};
        std::array<int,4> feet{},foot_geoms{};
        for(int f=0;f<4;++f){
            feet[f]=mj_name2id(m,mjOBJ_BODY,(legs[f]+"_FOOT").c_str());foot_geoms[f]=-1;
            for(int g=0;g<m->ngeom;++g)if(m->geom_bodyid[g]==feet[f]&&m->geom_type[g]==mjGEOM_SPHERE)foot_geoms[f]=g;
            if(foot_geoms[f]<0)throw std::runtime_error("Missing foot sphere");
        }
        mj_forward(m,d);
        double min_bottom=1e9;
        for(int f=0;f<4;++f)min_bottom=std::min(min_bottom,d->geom_xpos[3*foot_geoms[f]+2]-m->geom_size[3*foot_geoms[f]]);
        d->qpos[2]+=.005-min_bottom;mj_forward(m,d);
        const int control_stride=std::lround(control_dt/m->opt.timestep),policy_stride=std::lround(policy_dt/m->opt.timestep);
        if(control_stride<1||policy_stride<1||std::abs(policy_stride*m->opt.timestep-policy_dt)>1e-8)
            throw std::runtime_error("Non-integral simulation/control timestep ratio");
        std::ofstream csv(argv[5]);if(!csv)throw std::runtime_error("Cannot open output CSV");
        csv<<std::setprecision(10);
        csv<<"t_s,command_vx,command_vy,command_wz,x_m,y_m,z_m,vx_body,vy_body,vz_body,roll_rad,pitch_rad,yaw_rad,gyro_x_rads,gyro_y_rads,gyro_z_rads,motor_age_ms,imu_age_ms,action_saturated";
        for(const auto& n:names)csv<<","<<n;
        for(const auto& n:names)csv<<",dq_"<<n;
        for(const auto& n:names)csv<<",qdes_"<<n;
        for(const auto& n:names)csv<<",tau_"<<n;
        for(const auto& l:legs)for(const auto& n:{"contact_n","clearance_m","x_body","y_body","z_body","hip_y_body","toe_yaw_body","slip_mps"})csv<<","<<l<<"_"<<n;
        csv<<"\n";
        std::vector<float> target=q0,zero(12,0),last_action(12,0);
        int saturated=0,rows=0;std::string failure;
        {
            MujocoMotorPd pd;pd.Bind(m,mapping);pd.Publish(target,zero,fixed_kp,fixed_kd,zero);
            MujocoFeedback feedback;feedback.Configure(m,mapping,motor_delay,imu_delay);
            MujocoFeedback::State sample;
            const int nsteps=std::lround((settle_s+zero_policy_s+seconds)/m->opt.timestep);
            for(int step=0;step<nsteps;++step){
                const double elapsed=step*m->opt.timestep,t=elapsed-settle_s-zero_policy_s;
                feedback.Sample(m,d);bool ready=feedback.Read(d->time,sample);
                if(elapsed>=settle_s&&step%policy_stride==0&&ready){
                    std::vector<float> obs;obs.reserve(45);
                    for(double w:sample.imu.gyro)obs.push_back(w*ang_scale);
                    mjtNum inv[4]={sample.imu.quat[0],-sample.imu.quat[1],-sample.imu.quat[2],-sample.imu.quat[3]},gravity[3]={0,0,-1},projected[3];
                    mju_rotVecQuat(projected,gravity,inv);
                    for(double g:projected)obs.push_back(g);
                    for(int k=0;k<3;++k)obs.push_back((t>=0?command[k]:0)*cmd_scale[k]);
                    for(int k=0;k<12;++k)obs.push_back((sample.motors.q[k]-q0[k])*q_scale);
                    for(double dq:sample.motors.dq)obs.push_back(dq*dq_scale);
                    obs.insert(obs.end(),last_action.begin(),last_action.end());
                    for(auto& o:obs)o=std::clamp(o,-clip_obs,clip_obs);
                    auto input=torch::from_blob(obs.data(),{1,45},torch::kFloat32);
                    auto output=policy.forward({input}).toTensor().contiguous().to(torch::kCPU);
                    if(output.numel()!=12||!torch::isfinite(output).all().item<bool>())throw std::runtime_error("Invalid policy output");
                    auto* a=output.data_ptr<float>();saturated=0;
                    for(int k=0;k<12;++k){saturated+=(a[k]<=low[k]||a[k]>=high[k]);last_action[k]=std::clamp(a[k],low[k],high[k]);target[k]=q0[k]+scale[k]*last_action[k];}
                    pd.Publish(target,zero,kp,kd,zero);
                }
                mj_step(m,d);feedback.Sample(m,d);
                if(step%control_stride==0&&t>=0){
                    // MuJoCo BODY local axes use the principal-inertia frame.
                    // The IMU site uses the intended robot x-forward/y-left/z-up frame.
                    mjtNum velocity[6];mj_objectVelocity(m,d,mjOBJ_SITE,imu_id,velocity,0);
                    double vworld[3]={velocity[3],velocity[4],velocity[5]},vbody[3];BodyVector(d->xmat+9*base_id,vworld,vbody);
                    mjtNum gyro[6];mj_objectVelocity(m,d,mjOBJ_SITE,imu_id,gyro,1);
                    const auto rpy=Rpy(d->qpos+3);
                    csv<<t;for(double c:command)csv<<","<<c;
                    for(int k=0;k<3;++k)csv<<","<<d->xpos[3*base_id+k];
                    for(double v:vbody)csv<<","<<v;
                    for(double e:rpy)csv<<","<<e;
                    for(int k=0;k<3;++k)csv<<","<<gyro[k];
                    csv<<","<<(d->time-sample.motor_time)*1000<<","<<(d->time-sample.imu_time)*1000<<","<<saturated;
                    for(int k=0;k<12;++k)csv<<","<<d->qpos[qadr[k]];
                    for(int k=0;k<12;++k)csv<<","<<d->qvel[vadr[k]];
                    for(float q:target)csv<<","<<q;
                    for(int k=0;k<12;++k)csv<<","<<d->actuator_force[mapping[k]];
                    for(int f=0;f<4;++f){
                        double normal_force=0;
                        for(int c=0;c<d->ncon;++c){const auto& con=d->contact[c];
                            if((con.geom[0]==floor&&con.geom[1]==foot_geoms[f])||(con.geom[1]==floor&&con.geom[0]==foot_geoms[f])){
                                mjtNum force[6];mj_contactForce(m,d,c,force);normal_force+=std::max(0.,force[0]);
                            }
                        }
                        double delta[3],body[3],hipdelta[3],hipbody[3];
                        for(int k=0;k<3;++k){delta[k]=d->geom_xpos[3*foot_geoms[f]+k]-d->xpos[3*base_id+k];hipdelta[k]=d->xanchor[3*jids[3*f]+k]-d->xpos[3*base_id+k];}
                        BodyVector(d->xmat+9*base_id,delta,body);BodyVector(d->xmat+9*base_id,hipdelta,hipbody);
                        double toe_x[3]={d->xmat[9*feet[f]],d->xmat[9*feet[f]+3],d->xmat[9*feet[f]+6]},toe_body[3];BodyVector(d->xmat+9*base_id,toe_x,toe_body);
                        mjtNum footv[6];mj_objectVelocity(m,d,mjOBJ_GEOM,foot_geoms[f],footv,0);
                        csv<<","<<normal_force<<","<<d->geom_xpos[3*foot_geoms[f]+2]-m->geom_size[3*foot_geoms[f]];
                        for(double b:body)csv<<","<<b;
                        csv<<","<<hipbody[1]<<","<<std::atan2(toe_body[1],toe_body[0])<<","<<std::hypot(footv[3],footv[4]);
                    }
                    csv<<"\n";++rows;
                    if(std::abs(rpy[0])>1.0472||std::abs(rpy[1])>1.0472||d->xpos[3*base_id+2]<.12){failure="tilt_or_low_base";break;}
                }
            }
        }
        std::cout<<"{\"mujoco_version\":\""<<mj_versionString()<<"\",\"rows\":"<<rows<<",\"physics_dt\":"<<m->opt.timestep
                 <<",\"policy_dt\":"<<policy_dt<<",\"motor_delay_ms\":"<<motor_delay<<",\"imu_delay_ms\":"<<imu_delay
                 <<",\"failure\":\""<<failure<<"\"}"<<std::endl;
        mj_deleteData(d);mj_deleteModel(m);return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;if(d)mj_deleteData(d);if(m)mj_deleteModel(m);return 1;}
}
