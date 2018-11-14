#!/usr/bin/env bash

# ----- INSTRUCTION ----------------------------------------------------------#
# This script allows the download from google drive of some trained models of 
# neural convolution networks for image classifications.
# If you want to use the Intel Movidius computer stick you need to provide the
# first argument `movidius` (without quotes)
# Otherwise it is sufficient to pass a first fictitious parameter, eg 'a'
# As a second parameter it is necessary to pass the path of the file or folder
# containing the images to be submitted to the network.
#
#Example:
#
# - run on intel movidius and single file:
# sh nomescript.sh movidius ../test.jpeg
#
# - run on intel movidius and test images folder:
# sh nomescript.sh movidius ../test_folder
#
# - run on intel movidius and single file:
# sh nomescript.sh empty ../test.jpeg
#
# - run test host and folder images:
# sh nomescript.sh empty ../test_folder
#
# NB empty is a fake parameter
#

#---- FUNCTION DEFINITION ----------------------------------------------------#
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
rm $file

# return parent folder and get config and model file
cd ..
config_file=$(find -L "${PWD}/${dest_dir}" -name \*.json | sort)
if [ "$1" = "movidius" ]; then
	model_file=$(find -L "${PWD}/${dest_dir}" -name \*.graph | sort)
else
	model_file=$(find -L "${PWD}/${dest_dir}" -name \*.pb | sort)
fi
echo "Found: ${config_file}"
echo "Found: ${model_file}"

# download test folder contains images
echo "Download test images..."
remote_test_zip='16XMxZahk2OCETN-HiSVtIJkGqyNsbzJb'
zip_test_folder=$(downloadfromGDrive $remote_test_zip)
echo "decompressing ${zip_test_folder}..."
test_folder=TestImages
mkdir -p $test_folder && tar xf $zip_test_folder -C $test_folder --strip-components=1
rm $zip_test_folder

# launch script
echo "python3 prediction.py --configfile ${config_file} --model ${model_file} --test ${test_folder}"
python3 prediction.py --configfile ${config_file} --model ${model_file} --test ${test_folder}


