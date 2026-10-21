// Read-only Gazebo Classic contact telemetry. Build/run commands are recorded
// in docs/deployment/amp_engine_comparison_2026-10-14.md.
#include <gazebo/gazebo_client.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

std::atomic<bool> running{true};
double last_time = -1.0;
void stop(int) { running = false; }

void on_contacts(ConstContactsPtr &message)
{
    const double time = message->time().sec() + message->time().nsec() * 1e-9;
    if (last_time >= 0 && time - last_time < 0.0099) return;
    last_time = time;
    std::array<std::array<double, 3>, 4> feet{};
    std::array<int, 4> points{};
    double fx = 0, fy = 0, fz = 0, depth = 0;
    int other_points = 0, self_pairs = 0;
    const std::array<std::string, 4> legs = {"FL", "FR", "RL", "RR"};
    for (int i = 0; i < message->contact_size(); ++i)
    {
        const auto &contact = message->contact(i);
        const bool first = contact.collision1().find("imgo2_gazebo::") == 0;
        const bool second = contact.collision2().find("imgo2_gazebo::") == 0;
        if (first && second) { ++self_pairs; continue; }
        if (!first && !second) continue;
        const auto &name = first ? contact.collision1() : contact.collision2();
        const auto &other = first ? contact.collision2() : contact.collision1();
        if (other.find("ground_plane::") != 0) continue;
        int foot = -1;
        for (int j = 0; j < 4; ++j)
            if (name.find(legs[j] + "_FOOT") != std::string::npos) foot = j;
        for (int j = 0; j < contact.depth_size(); ++j)
            depth = std::max(depth, contact.depth(j));
        for (int j = 0; j < contact.wrench_size(); ++j)
        {
            const auto &wrench = contact.wrench(j);
            // ODE publishes each wrench in its LINK frame (not world).
            // Static-ground feedback is undefined; read only the robot side.
            // The offline tool rotates each FOOT-lumped SHANK force using FK.
            const bool body_first = wrench.body_1_name().find("imgo2_gazebo::") == 0;
            const auto &force = body_first ? wrench.body_1_wrench().force() : wrench.body_2_wrench().force();
            fx += force.x(); fy += force.y(); fz += force.z();
            if (foot >= 0) {
                feet[foot][0] += force.x(); feet[foot][1] += force.y(); feet[foot][2] += force.z();
                ++points[foot];
            } else ++other_points;
        }
    }
    std::cout << std::setprecision(17) << time << ',' << fx << ',' << fy << ',' << fz;
    for (int j = 0; j < 4; ++j)
        std::cout << ',' << feet[j][0] << ',' << feet[j][1] << ',' << feet[j][2] << ',' << points[j];
    std::cout << ',' << other_points << ',' << self_pairs << ',' << depth << std::endl;
}

int main(int argc, char **argv)
{
    std::signal(SIGINT, stop); std::signal(SIGTERM, stop);
    gazebo::client::setup(argc, argv);
    gazebo::transport::NodePtr node(new gazebo::transport::Node());
    node->Init();
    std::cout << "sim_s,sum_link_fx,sum_link_fy,sum_link_fz";
    for (const auto &leg : {"FL", "FR", "RL", "RR"})
        std::cout << ',' << leg << "_fx," << leg << "_fy," << leg << "_fz," << leg << "_points";
    std::cout << ",other_points,self_pairs,max_depth_m" << std::endl;
    auto subscriber = node->Subscribe("/gazebo/default/physics/contacts", on_contacts);
    while (running) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    subscriber.reset(); node->Fini(); gazebo::client::shutdown();
}
