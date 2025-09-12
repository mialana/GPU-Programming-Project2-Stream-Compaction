#!/usr/bin/env bash

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

while true; do
    read -p "Do you want to re-build? (y/n): " yn
    case $yn in
        [Yy]* )
        cd $SCRIPT_DIR; # cd to where script is located

        trash build;
        mkdir build && cd build;
        cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-I/usr/local/cuda-13.0/targets/x86_64-linux/include/cccl/thrust/;
        make -j$(nproc --all);

        cd $CURRENT_DIR; # cd to stored original directory 
        break;;
        [Nn]* ) 
        break;;
        * ) echo "Invalid input. Please answer 'y' or 'n'.";;
    esac
done

$SCRIPT_DIR/build/bin/cis5650_stream_compaction_test
