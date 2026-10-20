#include "my_myth.h"
namespace my_myth {
double quinticProgress(double t,double duration) {
    if(t<=0) return 0;
    if(t>=duration) return 1;
    const double tau=t/duration;
    return tau*tau*tau*(10.0+tau*(-15.0+6.0*tau));
}
double quinticRate(double t,double duration) {
    if(t<=0 || t>=duration) return 0;
    const double tau=t/duration;
    return 30.0*tau*tau*(1.0+tau*(-2.0+tau))/duration;
}
}

#include <cmath>
#include <stdexcept>
namespace my_myth {
std::array<float,3> projectedGravity(const std::array<float,4>& q) {
    double norm=0;for(float x:q) {if(!std::isfinite(x)) throw std::invalid_argument("nonfinite quaternion");norm+=double(x)*x;}
    if(norm<1e-12 || std::abs(norm-1.0)>0.01) throw std::invalid_argument("quaternion must be unit wxyz");
    const double w=q[0],x=q[1],y=q[2],z=q[3];
    return {{float(2*(w*y-x*z)),float(-2*(w*x+y*z)),float(-1+2*(x*x+y*y))}};
}
}
