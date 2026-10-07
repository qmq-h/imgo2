cd "D:\LXL0929\AAApostgraduate\SmallDog\imgo2\imgo2_description_real\xacro"

$env:PYTHONPATH="$env:TEMP\codex-xacro-deps"
// 生成原版urdf
python -X utf8 -c "import xacro; xacro.main()" robot.xacro -o ../urdf/imgo2_real.urdf
// 生成gazebo urdf
python -X utf8 -c "import xacro; xacro.main()" robot.xacro transmission:=true gazebo:=true imu:=true -o ../urdf/imgo2_real.gazebo.urdf