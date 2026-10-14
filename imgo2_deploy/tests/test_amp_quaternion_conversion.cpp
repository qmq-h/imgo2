// Check the deployment's real inverse rotation against independent Rodrigues R^T.
#include "../src/imgo2_deploy/library/core/vector_math/vector_math.hpp"
#include <array>
#include <iostream>
#include <random>
#include <stdexcept>

int main()
{
    std::mt19937 random(42);
    std::uniform_real_distribution<double> sample(-1.0, 1.0);
    double max_error = 0.0;
    for (int i = 0; i < 500; ++i)
    {
        std::array<double, 3> axis{sample(random), sample(random), sample(random)};
        const double norm = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
        for (double &value : axis) value /= norm;
        const double angle = sample(random) * 3.141592653589793;
        const double c = std::cos(angle), s = std::sin(angle);
        const std::vector<float> q{float(std::cos(angle/2)), float(axis[0]*std::sin(angle/2)),
                                  float(axis[1]*std::sin(angle/2)), float(axis[2]*std::sin(angle/2))};
        // Independent active rotation matrix, constructed from axis and angle.
        const double cross[3][3]{{0,-axis[2],axis[1]}, {axis[2],0,-axis[0]}, {-axis[1],axis[0],0}};
        double rotation[3][3];
        for (int row=0; row<3; ++row)
            for (int col=0; col<3; ++col)
                rotation[row][col] = c*(row==col) + (1-c)*axis[row]*axis[col] + s*cross[row][col];
        for (const auto &v : {std::vector<float>{0,0,-1},
                              std::vector<float>{float(sample(random)),float(sample(random)),float(sample(random))}})
        {
            const auto actual = QuatRotateInverse(q,v);
            for (int row=0; row<3; ++row)
            {
                double expected=0;
                for (int col=0; col<3; ++col) expected += rotation[col][row]*v[col];
                max_error = std::max(max_error, std::fabs(expected-actual[row]));
            }
        }
    }
    // ROS xyzw must be reordered to the helper's wxyz contract.
    const float h = std::sqrt(0.5f);
    const auto yaw = QuatRotateInverse({h,0,0,h},{1,0,0});
    const auto pitch = QuatRotateInverse({h,0,h,0},{0,0,-1});
    if (max_error>1e-6 || std::fabs(yaw[1]+1)>1e-6 || std::fabs(pitch[0]-1)>1e-6)
        throw std::runtime_error("Inverse rotation/wxyz convention mismatch");
    std::cout << "500 orientations, gravity and velocity: max error=" << max_error << '\n';
}
