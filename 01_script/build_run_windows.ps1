# Run on Visual Studio 2019 Developer PowerShell
# You may need the following command before executing this script
# Set-ExecutionPolicy Unrestricted -Scope Process

Param(
    [string]$FRAMEWORK_NAME = "MNN",
    [switch]$BUILD_ONLY
)
$LOG_HEADER = "[CI_WINDOWS_${FRAMEWORK_NAME}]"
echo "${LOG_HEADER} Start"

echo "${LOG_HEADER} Build Start"
if(Test-Path build) {
    del -R build
}
mkdir build
cd build
cmake -DINFERENCE_HELPER_ENABLE_"$FRAMEWORK_NAME"=on ../pj_cls_mobilenet_v2_wo_opencv
MSBuild -m:4 ./main.sln /p:Configuration=Release
if(!($?)) {
    echo "${LOG_HEADER} Build Error"
    cd ..
    exit -1
}
echo "${LOG_HEADER} Build End"

if($BUILD_ONLY) {
    cd ..
    exit 0
}


echo "${LOG_HEADER} Run Start"
./Release/main.exe
if(!($?)) {
    echo "${LOG_HEADER} Run Error"
    cd ..
    exit -1
}
echo "${LOG_HEADER} Run End"

cd ..
echo "$FRAMEWORK_NAME" >> time_inference_windows.txt
cat build/time_inference.txt >> time_inference_windows.txt

echo "${LOG_HEADER} End"

exit  0
