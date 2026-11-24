// Simulation-only camera plugins: follow translation and save native RGB frames.
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/sensors/CameraSensor.hh>
#include <gazebo/rendering/Camera.hh>
#include <gazebo/common/Image.hh>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstdio>
#include <jpeglib.h>

namespace gazebo {
#ifdef MEDIA_WORLD_PLUGIN
class MediaFollow : public WorldPlugin {
  physics::WorldPtr world;physics::ModelPtr camera;event::ConnectionPtr update;
  ignition::math::Pose3d offset{-1.45,-1.65,.75,0,.32,.85};
 public:void Load(physics::WorldPtr w,sdf::ElementPtr config) override {
    if(config->HasElement("offset")) offset=config->Get<ignition::math::Pose3d>("offset");
    world=w;update=event::Events::ConnectWorldUpdateBegin([this](const common::UpdateInfo&){
      auto robot=world->ModelByName("imgo2_gazebo");camera=world->ModelByName("defense_camera");
      if(robot&&camera){auto p=robot->WorldPose().Pos();
        camera->SetWorldPose(ignition::math::Pose3d(p+offset.Pos(),offset.Rot()));}
    });
  }
};
GZ_REGISTER_WORLD_PLUGIN(MediaFollow)
#else
class MediaFrames : public SensorPlugin {
  sensors::CameraSensorPtr sensor;event::ConnectionPtr frames;std::string output;std::ofstream times;unsigned int count=0;
 public:void Load(sensors::SensorPtr base,sdf::ElementPtr config) override {
    sensor=std::dynamic_pointer_cast<sensors::CameraSensor>(base);if(!sensor)gzthrow("expected camera");
    output=config->Get<std::string>("output");times.open(output+"/frames.csv");times<<"frame,sim_s\n";
    frames=sensor->Camera()->ConnectNewImageFrame(std::bind(&MediaFrames::OnFrame,this,std::placeholders::_1,std::placeholders::_2,std::placeholders::_3,std::placeholders::_4,std::placeholders::_5));
    sensor->SetActive(true);
  }
  void OnFrame(const unsigned char* pixels,unsigned int w,unsigned int h,unsigned int depth,const std::string&) {
    if(depth!=3)return;std::ostringstream name;name<<output<<"/frame_"<<std::setw(6)<<std::setfill('0')<<count;
    if(w<=1920){
      name<<".jpg";FILE* file=std::fopen(name.str().c_str(),"wb");if(!file)return;
      jpeg_compress_struct compressor;jpeg_error_mgr errors;compressor.err=jpeg_std_error(&errors);jpeg_create_compress(&compressor);jpeg_stdio_dest(&compressor,file);
      compressor.image_width=w;compressor.image_height=h;compressor.input_components=3;compressor.in_color_space=JCS_RGB;
      jpeg_set_defaults(&compressor);jpeg_set_quality(&compressor,95,TRUE);jpeg_start_compress(&compressor,TRUE);
      while(compressor.next_scanline<h){JSAMPROW row=const_cast<unsigned char*>(pixels+compressor.next_scanline*w*3);jpeg_write_scanlines(&compressor,&row,1);}
      jpeg_finish_compress(&compressor);jpeg_destroy_compress(&compressor);std::fclose(file);
    }else{
      name<<".png";common::Image image;image.SetFromData(pixels,w,h,common::Image::RGB_INT8);image.SavePNG(name.str());
    }
    times<<count++<<","<<std::setprecision(12)<<sensor->LastMeasurementTime().Double()<<"\n";times.flush();
  }
};
GZ_REGISTER_SENSOR_PLUGIN(MediaFrames)
#endif
}
