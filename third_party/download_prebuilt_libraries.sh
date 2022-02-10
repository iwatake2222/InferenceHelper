move_dir_to_shell_file() {
    dir_shell_file=`dirname "$0"`
    cd ${dir_shell_file}
}

download_and_extract_ncnn() {
    local url_base="https://github.com/Tencent/ncnn/releases/download/"
    local tag="20211208"
    local prefix="ncnn-20211208-"
    local name=$1
    local url="${url_base}${tag}/${prefix}${name}.zip"
    echo "Downloading ${url}"
    curl -Lo temp.zip  ${url}
    unzip -o temp.zip
    rm -rf ${name}
    mv ${prefix}${name} ${name}
}

download_and_extract() {
    local url=$1
    echo "Downloading ${url}"
    local ext=${url##*.}
    if [ `echo ${ext} | grep zip` ]; then
        curl -Lo temp.zip ${url}
        unzip -o temp.zip
        rm temp.zip
    else
        curl -Lo temp.tgz ${url}
        tar xzvf temp.tgz
        rm temp.tgz
    fi
}
########################################################################

### cd to the same directory as this shell file ###
move_dir_to_shell_file

### Get other projects (dependencies) ###
git submodule update --init --recommend-shallow --depth 1

### Download pre-built libraries from https://github.com/iwatake2222/InferenceHelper_Binary ###
download_and_extract "https://github.com/iwatake2222/InferenceHelper_Binary/releases/download/v20220210/armnn_prebuilt_linux_1804.tgz"
# download_and_extract "https://github.com/iwatake2222/InferenceHelper_Binary/releases/download/v20220210/armnn_prebuilt_linux_2004.tgz"
download_and_extract "https://github.com/iwatake2222/InferenceHelper_Binary/releases/download/v20220210/edgetpu_prebuilt_linux_1804.tgz"
# download_and_extract "https://github.com/iwatake2222/InferenceHelper_Binary/releases/download/v20220210/edgetpu_prebuilt_linux_2004.tgz"
download_and_extract "https://github.com/iwatake2222/InferenceHelper_Binary/releases/download/v20220210/edgetpu_prebuilt_windows.zip"
download_and_extract "https://github.com/iwatake2222/InferenceHelper_Binary/releases/download/v20220210/mnn_prebuilt_linux_1804.tgz"
# download_and_extract "https://github.com/iwatake2222/InferenceHelper_Binary/releases/download/v20220210/mnn_prebuilt_linux_2004.tgz"
download_and_extract "https://github.com/iwatake2222/InferenceHelper_Binary/releases/download/v20220210/mnn_prebuilt_windows.zip"
download_and_extract "https://github.com/iwatake2222/InferenceHelper_Binary/releases/download/v20220210/tflite_prebuilt_linux_1804.tgz"
# download_and_extract "https://github.com/iwatake2222/InferenceHelper_Binary/releases/download/v20220210/tflite_prebuilt_linux_2004.tgz"
download_and_extract "https://github.com/iwatake2222/InferenceHelper_Binary/releases/download/v20220210/tflite_prebuilt_windows.zip"

### Download ncnn pre-built libraries from https://github.com/Tencent/ncnn ###
mkdir -p ncnn_prebuilt && cd ncnn_prebuilt
download_and_extract_ncnn "android-vulkan"
download_and_extract_ncnn "android"
download_and_extract_ncnn "ubuntu-2004-shared"
download_and_extract_ncnn "ubuntu-2004"
download_and_extract_ncnn "windows-vs2019-shared"
download_and_extract_ncnn "windows-vs2019"
cd ..


### Download NNabla pre-built libraries from https://github.com/iwatake2222/InferenceHelper
mkdir -p nnabla_prebuilt/windows-vs2019 && mkdir -p nnabla_prebuilt/aarch64 && cd nnabla_prebuilt
curl -L https://nnabla.org/cpplib/1.25.0/nnabla-cpplib-1.25.0-win64.zip -o temp.zip
unzip -o temp.zip
rm temp.zip
mv nnabla-cpplib-1.25.0-win64/* windows-vs2019/.

# curl -L https://nnabla.org/cpplib/1.25.0/nnabla-cpplib-cuda_110_8-1.25.0-win64.zip -o temp.zip
# unzip -o temp.zip
# rm temp.zip
# mv nnabla-cpplib-cuda_110_8-1.25.0-win64/* windows-vs2019/.

curl -L https://github.com/libarchive/libarchive/releases/download/v3.5.2/libarchive-v3.5.2-win64.zip -o temp.zip
unzip -o temp.zip
mv libarchive/bin/archive.dll windows-vs2019/bin/.

curl -L https://nnabla.org/cpplib/1.25.0/nnabla-cpplib-1.25.0-Linux_aarch64.zip -o temp.zip
unzip -o temp.zip
rm temp.zip
mv nnabla-cpplib-1.25.0-Linux_aarch64/* aarch64/.
