#include "mujoco_motor_pd.hpp"
#include <array>
#include <iostream>
#include <thread>
#include <atomic>
#include <numeric>
#include <limits>

void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
int main(int argc, char** argv)
{
    try {
        char error[2048]{};
        mjModel* m = mj_loadXML(argv[1], nullptr, error, sizeof(error));
        check(m, error);
        mjData* d = mj_makeData(m);
        std::vector<int> map(12); std::iota(map.begin(), map.end(), 0);
        std::vector<double> zero(12, 0), one(12, 1), kd(12, 8);
        double passive_rms = 0;
        {
            MujocoMotorPd pd;
            pd.Bind(m, map);
            pd.Publish(zero, zero, zero, kd, zero);
            int joint = m->actuator_trnid[0], v = m->jnt_dofadr[joint];
            d->qvel[v] = 1;
            mj_forward(m, d);
            check(std::abs(d->ctrl[0] + 8) < 1e-12, "first braking torque");
            d->qvel[v] = -1; // no new host command
            mj_forward(m, d);
            check(std::abs(d->ctrl[0] - 8) < 1e-12, "PD must use new velocity without republish");
            auto huge = std::vector<double>(12, 100);
            pd.Publish(zero, zero, zero, zero, huge);
            mj_forward(m, d);
            check(d->ctrl[0] == 23.7, "physical torque clamp");
            bool rejected = false;
            auto bad = zero; bad[11] = std::numeric_limits<double>::quiet_NaN();
            try { pd.Publish(bad, zero, zero, zero, zero); } catch (const std::invalid_argument&) { rejected = true; }
            check(rejected, "nonfinite rejection");
            mj_forward(m, d);
            check(d->ctrl[0] == 23.7, "invalid command must not partially publish");
            mj_resetData(m, d);
            pd.Publish(zero, zero, zero, kd, zero);
            double sum = 0; int samples = 0;
            for (int i = 0; i < 10000; ++i) {
                mj_step(m, d);
                if (i >= 8000) for (int a = 0; a < 12; ++a) {
                    double speed = d->qvel[m->jnt_dofadr[m->actuator_trnid[2*a]]];
                    sum += speed*speed; ++samples;
                }
            }
            passive_rms = std::sqrt(sum/samples);
            check(passive_rms < .2, "Passive sustained shaking regression");
            for (int w = 0; w < mjNWARNING; ++w) check(d->warning[w].number == 0, "physics warning");
            // Reset and concurrent publication must not expose a mixed command.
            mj_resetData(m, d);
            std::atomic<bool> stop{false};
            std::thread writer([&] { while (!stop) { pd.Publish(zero,zero,zero,zero,zero); pd.Publish(zero,zero,zero,zero,one); } });
            bool coherent = true;
            for (int i = 0; i < 1000; ++i) {
                mj_forward(m, d);
                for (int a = 1; a < 12; ++a) coherent &= d->ctrl[a] == d->ctrl[0];
            }
            stop = true; writer.join(); check(coherent, "atomic whole-command publication");
            // A second model is ignored until explicitly rebound (reload/compile).
            mjModel* m2 = mj_loadXML(argv[1], nullptr, error, sizeof(error)); check(m2, error);
            mjData* d2 = mj_makeData(m2); mj_forward(m2, d2);
            check(d2->ctrl[0] == 0, "foreign model mapping used");
            pd.Bind(m2, map); pd.Publish(zero,zero,zero,zero,one); mj_forward(m2,d2);
            check(d2->ctrl[0] == 1, "model rebind");
            mj_deleteData(d2); mj_deleteModel(m2);
        }
        check(mjcb_control == nullptr, "callback uninstall");
        { MujocoMotorPd second; second.Bind(m,map); }
        mj_deleteData(d); mj_deleteModel(m);
        std::cout << "per-step PD / clamp / atomic command / reset / rebind / lifetime PASS; Passive rms=" << passive_rms << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
