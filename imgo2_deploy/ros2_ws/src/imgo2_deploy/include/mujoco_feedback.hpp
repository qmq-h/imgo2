#pragma once
#include <mujoco/mujoco.h>
#include <array>
#include <cmath>
#include <deque>
#include <random>
#include <stdexcept>
#include <vector>

// Called under the simulation mutex. Source samples use simulation time, so
// pause/slow motion does not age data or manufacture new noise samples.
class MujocoFeedback
{
public:
    struct Motors { std::array<double,12> q{}, dq{}, tau{}; };
    struct Imu { std::array<double,4> quat{1,0,0,0}; std::array<double,3> gyro{}; };
    struct State { Motors motors; Imu imu; double motor_time=0, imu_time=0; };

    void Configure(const mjModel* m, const std::vector<int>& mapping, double motor_ms, double imu_ms)
    {
        ValidateDelay(motor_ms); ValidateDelay(imu_ms);
        if (mapping.size()!=12 || m->nsensordata<43) throw std::invalid_argument("Incompatible MuJoCo feedback layout");
        for (int i=0;i<12;++i) {
            if(mapping[i]<0 || mapping[i]>=12) throw std::invalid_argument("Invalid feedback mapping");
            mapping_[i]=mapping[i];
        }
        const int quat=mj_name2id(m,mjOBJ_SENSOR,"orientation"), gyro=mj_name2id(m,mjOBJ_SENSOR,"angular_velocity");
        if(quat<0 || gyro<0 || m->sensor_dim[quat]!=4 || m->sensor_dim[gyro]!=3)
            throw std::invalid_argument("Missing MuJoCo IMU sensors");
        quat_adr_=m->sensor_adr[quat]; gyro_adr_=m->sensor_adr[gyro];
        angle_std_=m->sensor_noise[quat]; gyro_std_=m->sensor_noise[gyro];
        if(!std::isfinite(angle_std_) || !std::isfinite(gyro_std_) || angle_std_<0 || gyro_std_<0)
            throw std::invalid_argument("IMU noise standard deviations must be finite and nonnegative");
        const double hz=Numeric(m,"imu_sample_rate_hz",200), seed=Numeric(m,"imu_noise_seed",12345);
        if(!std::isfinite(hz) || hz<=0 || hz>1/m->opt.timestep+1e-8 || !std::isfinite(seed) || seed<0 || seed>4294967295. || std::floor(seed)!=seed)
            throw std::invalid_argument("Invalid IMU sample rate or noise seed");
        motor_delay_=motor_ms*.001; imu_delay_=imu_ms*.001;
        imu_period_=1/hz; seed_=static_cast<unsigned>(seed); model_=m;
        Reset();
    }
    void Reset()
    {
        motors_.clear(); imus_.clear(); last_time_=-1; next_imu_=0;
        random_.seed(seed_); gaussian_.reset();
    }
    void Sample(const mjModel* m, const mjData* d)
    {
        if(m!=model_) return;
        if(d->time<last_time_) Reset();
        if(d->time<=last_time_) return; // no resampling while paused or polling
        last_time_=d->time;
        Motors motor;
        for(int i=0;i<12;++i){const int a=mapping_[i];motor.q[i]=d->sensordata[a];motor.dq[i]=d->sensordata[12+a];motor.tau[i]=d->sensordata[24+a];}
        motors_.push_back({d->time,motor}); Trim(motors_, d->time-motor_delay_);
        if(d->time+1e-12>=next_imu_) {
            Imu imu;
            for(int i=0;i<4;++i) imu.quat[i]=d->sensordata[quat_adr_+i];
            for(int i=0;i<3;++i) imu.gyro[i]=d->sensordata[gyro_adr_+i]+gyro_std_*gaussian_(random_);
            // Isotropic Gaussian rotation-vector perturbation in the IMU/body
            // frame. Never add four independent offsets to a unit quaternion.
            mjtNum axis[3];double angle=0;
            for(int i=0;i<3;++i){axis[i]=angle_std_*gaussian_(random_);angle+=axis[i]*axis[i];}
            angle=std::sqrt(angle);
            if(angle>0){for(double& a:axis)a/=angle;mjtNum delta[4], out[4];mju_axisAngle2Quat(delta,axis,angle);mju_mulQuat(out,imu.quat.data(),delta);for(int i=0;i<4;++i)imu.quat[i]=out[i];}
            mju_normalize4(imu.quat.data());
            imus_.push_back({d->time,imu}); Trim(imus_,d->time-imu_delay_);
            next_imu_=(std::floor((d->time+1e-12)/imu_period_)+1)*imu_period_;
        }
    }
    bool Read(double now, State& out)
    {
        Trim(motors_,now-motor_delay_);Trim(imus_,now-imu_delay_);
        if(motors_.empty() || imus_.empty() || motors_.front().time>now-motor_delay_+1e-12 || imus_.front().time>now-imu_delay_+1e-12) return false;
        out={motors_.front().value,imus_.front().value,motors_.front().time,imus_.front().time};
        return true;
    }
private:
    template<class T> struct Sampled { double time; T value; };
    template<class T> static void Trim(std::deque<Sampled<T>>& q,double cutoff)
    { while(q.size()>1 && q[1].time<=cutoff+1e-12)q.pop_front(); }
    static void ValidateDelay(double ms)
    {if(!std::isfinite(ms)||ms<0||ms>1000)throw std::invalid_argument("MuJoCo feedback delay must be in [0,1000] ms");}
    static double Numeric(const mjModel* m,const char* name,double fallback)
    {int id=mj_name2id(m,mjOBJ_NUMERIC,name);if(id<0)return fallback;if(m->numeric_size[id]!=1)throw std::invalid_argument("IMU numeric must be scalar");return m->numeric_data[m->numeric_adr[id]];}
    const mjModel* model_=nullptr;
    std::array<int,12> mapping_{};
    int quat_adr_=36,gyro_adr_=40;
    double motor_delay_=0,imu_delay_=0,imu_period_=.005,gyro_std_=0,angle_std_=0,last_time_=-1,next_imu_=0;
    unsigned seed_=12345;
    std::mt19937 random_{12345};std::normal_distribution<double> gaussian_{0,1};
    std::deque<Sampled<Motors>> motors_;
    std::deque<Sampled<Imu>> imus_;
};
