#include "go_trajectory.h"
#include "function/my_myth.h"
#include <cmath>
#include <stdexcept>
namespace go {
SmoothTrajectory::SmoothTrajectory(double duration):duration_(duration) {
    if(!std::isfinite(duration) || duration<=0) throw std::invalid_argument("SmoothTrajectory duration must be finite and greater than zero");
}
double SmoothTrajectory::progress(double t) const {return my_myth::quinticProgress(t,duration_);}
double SmoothTrajectory::rate(double t) const {return my_myth::quinticRate(t,duration_);}
}
