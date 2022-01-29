download_and_extract() {
    local url_base="https://github.com/Tencent/ncnn/releases/download/"
    local tag="20211208"
    local prefix="ncnn-20211208-"
    local name=$1
    local url="${url_base}${tag}/${prefix}${name}.zip"
    echo "Downloading ${url}"
    curl -Lo temp.zip  ${url}
    unzip temp.zip
    rm -rf ${name}
    mv ${prefix}${name} ${name}
}

move_dir_to_shell_file() {
    dir_shell_file=`dirname "$0"`
    cd ${dir_shell_file}
}

move_dir_to_shell_file
mkdir -p ncnn_prebuilt && cd ncnn_prebuilt

download_and_extract "android-vulkan"
download_and_extract "android"
download_and_extract "ubuntu-2004-shared"
download_and_extract "ubuntu-2004"
download_and_extract "windows-vs2019-shared"
download_and_extract "windows-vs2019"
