#!/bin/bash

# get the directory of the current script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# if the first argument is "clean", delete the build directory
if [ "$1" == "clean" ]; then
    rm -rf "$DIR/build"
fi

# create the build directory and navigate into it
mkdir -p "$DIR/build" && cd "$DIR/build"

# run cmake and make
cmake ..
make