#!/usr/bin/env bash
#----------------------------------------------------------------------------#
# define function to download the trained model from google drive
# 1st argument: must be the ID of file in GoogleDrive
# 2nd argument: destination folder
downloadfromGDrive() {
	ggID=$1
	ggURL='https://drive.google.com/uc?export=download'  
	filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"  
	getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"  
	curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${filename}"
	echo $filename  
}
#-----------------------------------------------------------------------------#


# display in which folder work
echo  "working directory:" $PWD

# dowload where download all file
dest_dir=Model
if [ ! -d "${dest_dir}" ]; then
	echo "Make folder . . ."
	mkdir $dest_dir
fi
cd $dest_dir

# download original file
if [ "$1" = "movidius" ]; then
	echo "Start download of model specific for Intel Movidius"
	model_file='1Cssy9EP8CSB6kf0tvV46o-518aLzqxYS' 
 	file=$(downloadfromGDrive $model_file $destdir)
else
	echo "Start download of model"
	model_file='1PLf5rLGPuuT5oQ_VS5kaWyLkiz4yOAen' 
	file=$(downloadfromGDrive $model_file $destdir)
fi
# decompress file
echo "decompressing file ${file}"
zip_files="${dest_dir}/${file}"
tar -zxvf $file 
# return parent folder and get config and model file
cd ..
config_file=$(find -L "${PWD}/${dest_dir}" -name \*.json | sort)
model_file=$(find -L "${PWD}/${dest_dir}" -name \*.graph | sort)
echo "Found: ${config_file}"
echo "Found: ${model_file}"

# launch script
echo "python3 prediction.py --configfile $config_file --model $model_file --test $2"
