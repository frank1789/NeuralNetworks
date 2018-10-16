echo "===== Prepare installation For Intel Nerual Compute Stick ====="
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y build-essential cmake pkg-config git libusb-dev
sudo apt-get install -y libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y libgtk2.0-dev libgtk-3-dev
sudo apt-get install -y libatlas-base-dev gfortran
sudo apt-get install -y libqtgui4 git gcc
sudo apt autoremove
echo "!!!! A T T E N T I O N !!!!"
echo "You will likely need to increase the swapfile size on the Raspberry Pi in
 order to successfully complete NCSDK and/or OpenCV installation.
 To increase the swapfile size, edit the value of 'CONF_SWAPSIZE'
 in /etc/dphys-swapfile"
 echo "The default value is 100 (MB). We recommend that you change this to
 1024 (MB) or greater."
 echo "Suggest: CONF_SWAPSIZE=2048"
 read -p "Press enter to continue"
sudo nano /etc/dphys-swapfile
echo "Then restart the swapfile service"
sudo /etc/init.d/dphys-swapfile restart
echo "===== Install Intel Sdk ====="
cd
git clone -b ncsdk2 https://github.com/movidius/ncsdk.git
cd ncsdk
make install
cd
echo "===== Start download OpenCV ====="
cd
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.4.3.zip
unzip opencv.zip
wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.4.3.zip
unzip opencv_contrib.zip
cd opencv-3.4.3
if [ -e CMakeCache.txt ]
then
    echo "file exist then remove"
		rm CMakeCache.txt
		echo "removed"
fi
mkdir build && cd cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
 -D CMAKE_INSTALL_PREFIX=/usr/local \
 -D BUILD_DOCS=OFF \
 -D BUILD_EXAMPLES=OFF\
 -D BUILD_TESTS=OFF\
 -D BUILD_opencv_ts=OFF\
 -D BUILD_PERF_TESTS=OFF\
 -D INSTALL_C_EXAMPLES=ON\
 -D INSTALL_PYTHON_EXAMPLES=ON\
 -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.3/modules \
 -D ENABLE_NEON=ON-D WITH_LIBV4L=ON
make -j4
sudo make install
sudo ldconfig
cd
pip3 install opencv-python
