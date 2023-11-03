#!/bin/bash

# Script requires the following tools:
#
# https://github.com/mgnauck/wgslminify
# https://github.com/mgnauck/js-payload-compress
# https://github.com/terser/terser 
#
# Run: ./compress.sh player.js C,V,F

if [[ $# -ne 2 ]]; then
  echo "Usage: ./compress input.js shader,function,names,excluded,from,mangling"
  exit 1
fi

function utilExists() {
  command -v $1 > /dev/null 2>&1 || { echo >&2 "Utility $1 is required to run compress script"; exit 1; }
}

utilExists wgslminify
utilExists js-payload-compress
#utilExists terser

infile=$1
infile_ext="${infile##*.}"
infile_name="${infile%.*}"
output_dir=output
shader_excludes=$2

# Replaces a multi-line string contained in one file into another file. Replace area is delimited by begin and end tag.
# Args: file_with_replace_targets begin_replace_tag end_replace_tag file_with_replace_content output_file
function replaceInFile() {
  echo -e "/$2/r $4\n/$2/,/$2/+1j\n/$3/-1,/$3/j\ns/$3//g\n/$2/s///g\nw $5\nq" | ed $1 > /dev/null
}

rm -rf $output_dir
mkdir $output_dir

wgslminify -e $shader_excludes visual.wgsl > $output_dir/visual_minified.wgsl

cd $output_dir

replaceInFile ../$infile BEGIN_VISUAL_SHADER END_VISUAL_SHADER visual_minified.wgsl ${infile_name}_with_shaders.${infile_ext}

#terser ${infile_name}_with_shaders.${infile_ext} -m -c toplevel,passes=5,drop_console=true,unsafe=true,pure_getters=true,keep_fargs=false,booleans_as_integers=true --toplevel > ${infile_name}_minified.${infile_ext}
cp ${infile_name}_with_shaders.${infile_ext} ${infile_name}_minified.${infile_ext}

js-payload-compress --zopfli-iterations=100 ${infile_name}_minified.${infile_ext} ${infile_name}_compressed.html
