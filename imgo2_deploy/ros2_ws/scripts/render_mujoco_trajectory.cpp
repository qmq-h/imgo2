// Native MuJoCo rendering of recorded closed-loop states. No policy export or physics edits.
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> out; std::stringstream stream(line); std::string part;
  while (std::getline(stream,part,',')) out.push_back(part); return out;
}
int main(int argc,char** argv) {
  if(argc!=10){std::cerr<<"scene csv width height fps start end azimuth distance\n";return 2;}
  try {
    const int width=std::stoi(argv[3]),height=std::stoi(argv[4]);
    const double fps=std::stod(argv[5]),start=std::stod(argv[6]),end=std::stod(argv[7]);
    if(width<1||height<1||fps<=0||end<start) throw std::runtime_error("invalid render dimensions/time");
    std::ifstream file(argv[2]);std::string line;std::getline(file,line);auto header=split(line);
    auto col=[&](const std::string& name){auto it=std::find(header.begin(),header.end(),name);
      if(it==header.end())throw std::runtime_error("missing CSV column "+name);return int(it-header.begin());};
    std::vector<std::string> names;
    for(auto leg:{"FL","FR","RL","RR"}) for(auto joint:{"hip","thigh","shank"}) names.push_back(std::string(leg)+"_"+joint+"_joint");
    std::vector<int> columns={col("t_s"),col("x_m"),col("y_m"),col("z_m"),col("roll_rad"),col("pitch_rad"),col("yaw_rad")};
    for(auto& name:names)columns.push_back(col(name));
    std::vector<std::array<double,19>> states;
    while(std::getline(file,line)){auto parts=split(line);std::array<double,19> state{};
      for(int i=0;i<19;++i)state[i]=std::stod(parts.at(columns[i]));states.push_back(state);}
    if(states.empty()||start<states.front()[0]-1e-5||end>states.back()[0]+1e-5) throw std::runtime_error("trajectory does not cover render interval");
    char error[2048]{};mjModel* model=mj_loadXML(argv[1],nullptr,error,sizeof(error));
    if(!model)throw std::runtime_error(error);mjData* data=mj_makeData(model);
    model->vis.global.offwidth=width;model->vis.global.offheight=height;
    if(!glfwInit())throw std::runtime_error("GLFW initialization failed");
    glfwWindowHint(GLFW_VISIBLE,GLFW_FALSE);glfwWindowHint(GLFW_SAMPLES,4);
    GLFWwindow* window=glfwCreateWindow(64,64,"Imgo2 native capture",nullptr,nullptr);
    if(!window)throw std::runtime_error("OpenGL context failed");glfwMakeContextCurrent(window);
    mjvCamera camera;mjv_defaultCamera(&camera);camera.azimuth=std::stod(argv[8]);camera.elevation=-19;camera.distance=std::stod(argv[9]);
    mjvOption option;mjv_defaultOption(&option);option.geomgroup[3]=0;option.sitegroup[0]=0;
    mjvScene scene;mjv_defaultScene(&scene);mjv_makeScene(model,&scene,5000);
    mjrContext context;mjr_defaultContext(&context);mjr_makeContext(model,&context,mjFONTSCALE_150);mjr_setBuffer(mjFB_OFFSCREEN,&context);
    std::vector<unsigned char> pixels(width*height*3);mjrRect viewport={0,0,width,height};
    std::array<int,12> qadr{};for(int i=0;i<12;++i){int id=mj_name2id(model,mjOBJ_JOINT,names[i].c_str());if(id<0)throw std::runtime_error("missing joint");qadr[i]=model->jnt_qposadr[id];}
    int frames=end==start?1:int(std::floor((end-start)*fps))+1;size_t cursor=0;
    for(int frame=0;frame<frames;++frame){double time=start+frame/fps;
      while(cursor+1<states.size()&&states[cursor+1][0]<=time)++cursor;
      auto a=states[cursor],b=states[std::min(cursor+1,states.size()-1)];
      double mix=b[0]>a[0]?std::clamp((time-a[0])/(b[0]-a[0]),0.,1.):0;
      std::array<double,19> value{};for(int j=0;j<19;++j)value[j]=a[j]+mix*(b[j]-a[j]);
      for(int j=0;j<3;++j)data->qpos[j]=value[1+j];
      double r=value[4]/2,p=value[5]/2,y=value[6]/2;
      data->qpos[3]=cos(r)*cos(p)*cos(y)+sin(r)*sin(p)*sin(y);
      data->qpos[4]=sin(r)*cos(p)*cos(y)-cos(r)*sin(p)*sin(y);
      data->qpos[5]=cos(r)*sin(p)*cos(y)+sin(r)*cos(p)*sin(y);
      data->qpos[6]=cos(r)*cos(p)*sin(y)-sin(r)*sin(p)*cos(y);
      for(int j=0;j<12;++j)data->qpos[qadr[j]]=value[7+j];mj_forward(model,data);
      camera.lookat[0]=value[1]+.08;camera.lookat[1]=value[2];camera.lookat[2]=value[3]-.08;
      mjv_updateScene(model,data,&option,nullptr,&camera,mjCAT_ALL,&scene);mjr_render(viewport,&scene,&context);
      mjr_readPixels(pixels.data(),nullptr,viewport,&context);
      for(int row=height-1;row>=0;--row)std::cout.write(reinterpret_cast<char*>(pixels.data()+row*width*3),width*3);
      if(frame%150==0)std::cerr<<"rendered "<<frame<<"/"<<frames<<"\n";
    }
    std::cout.flush();mjr_freeContext(&context);mjv_freeScene(&scene);mj_deleteData(data);mj_deleteModel(model);glfwDestroyWindow(window);glfwTerminate();
  }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
