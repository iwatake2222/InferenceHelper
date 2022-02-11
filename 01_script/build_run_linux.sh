# docker create  -v /mnt/c/iwatake/devel:/root/devel -v /etc/localtime:/etc/localtime:ro -it --name=ubuntu20 ubuntu:20.04
# docker start ubuntu20
# docker exec -it ubuntu20 bash

# Check if sudo needed
sudo
if [ "$?" -le 10 ]
then
L_SUDO=sudo
else
L_SUDO=
fi

set -e

FRAMEWORK_NAME=${1:-"MNN"}
BUILD_ONLY=${2:-0}
LOG_HEADER="[CI_LINUX_${FRAMEWORK_NAME}]"
echo "${LOG_HEADER} Start"

# ${L_SUDO} apt update
# ${L_SUDO} apt install -y g++ git cmake wget unzip vulkan-utils libvulkan1 libvulkan-dev


echo "${LOG_HEADER} Build Start"
cd pj_cls_mobilenet_v2_wo_opencv
rm -rf build
mkdir build && cd build
cmake .. -DINFERENCE_HELPER_ENABLE_${FRAMEWORK_NAME}=on
make -j4
echo "${LOG_HEADER} Build End"

if [ ${BUILD_ONLY} -ne 0 ]; then
    exit 0
fi

echo "${LOG_HEADER} Run Start"
./main
# if [ ${?} -ne 0 ]; then
#     echo "${LOG_HEADER} Run Error"
#     exit 1
# fi
echo "${LOG_HEADER} Run End"


echo "$FRAMEWORK_NAME" >> ../../time_inference_linux.txt
cat time_inference.txt >> ../../time_inference_linux.txt

echo "${LOG_HEADER} End"
