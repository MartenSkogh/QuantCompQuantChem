#!/usr/bin/env sh
###############################################################################
#      Author: Mårten Skogh
#        Date: 2019-04-03
# Description: Downloads the Qiskit repos from Github and installs them in the 
#              current directory.
################################################################################

#Clone all the Qiskit repos and build locally

#Check if we have all commands that we are going to use
if [ "$(command -v python3)" ] ; then
    echo "Found python3!" #Do nothing 
elif [ "$(command -v python3.5)" ] ; then 
    alias python3='python3.5' 
    echo "Aliased 'python3' to 'python3.7'" 
elif [ "$(command -v python3.6)" ] ; then 
    alias python3='python3.6' 
    echo "Aliased 'python3' to 'python3.7'" 
elif [ "$(command -v python3.7)" ] ; then 
    alias python3='python3.7' 
    echo "Aliased 'python3' to 'python3.7'" 
else
    echo "Could not find correct version of python"
    exit 1
fi

if [ "$(command -v pip3)" ] ; then
    echo "Found pip3!" #Do nothing 
elif [ "$(command -v pip3.5)" ] ; then 
    alias pip3='pip3.5' 
    echo "Aliased 'pip3' to 'pip3.5'" 
elif [ "$(command -v pip3.6)" ] ; then 
    alias pip3='pip3.6' 
    echo "Aliased 'pip3' to 'pip3.6'" 
elif [ "$(command -v pip3.7)" ] ; then 
    alias pip3='pip3.7' 
    echo "Aliased 'pip3' to 'pip3.7'" 
else
    echo "Could not find correct version of pip"
    exit 1
fi

if [ ! "$(command -v rsync)"] ; then
    echo "ERROR: Missing command 'rsync' required for installation!"
    exit 1
fi

QISKIT_PACKAGES="qiskit-terra qiskit-aer qiskit-ignis qiskit-ibmq-provider qiskit-aqua qiskit-chemistry"
python_version=$(python3 --version)

echo "Installing packages: "${QISKIT_PACKAGES[*]}

echo "Cloning Github repos..."
for package in ${QISKIT_PACKAGES[*]}
do
    if [ -d $package ] ; then 
        echo "Directory '$(pwd)/$package' already exists"
        echo "Removing directory $(pwd)/$package..."
        rm -rf $package
    fi

    git clone https://github.com/qiskit/$package

done

# Make sure all packeges are uninstalled through pip
echo "Removing Qiskit packages installed with pip..."
pip3 uninstall ${QISKIT_PACKAGES[*]}

# Run the installation
echo "Running install for packages..."
for package in ${QISKIT_PACKAGES[*]}
do
    cd ./$package 

    if [ -e requirements.txt ] ; then
        pip3 install -U -r requirements.txt
    fi
    
    if [ -e requirements-dev.txt ] ; then 
        pip3 install -U -r requirements-dev.txt
    fi

    python3 setup.py install
    
    cd ..
done

# Create a main dorectory
if [ ! -d qiskit ] ; then 
    mkdir qiskit
else
    rm -rf qiskit
    mkdir qiskit
fi 

echo "Creating single folder..."
for package in ${QISKIT_PACKAGES[*]}
do
    if [ -d ./$package/qiskit ] ; then
        rsync -av ./$package/qiskit .
    else
        echo "ERROR: Directory $(pwd)$package/qiskit not found!"
        exit 1
    fi

    if [ $package = "qiskit-terra" ] ; then
        rsync -av ./qiskit-terra/build/lib.linux-x86_64-${python_version:7:3}/qiskit .
    fi

    if [ $package = "qiskit-aer" ] ; then # Unsure if these need to be moved, but saftey first
        rsync -av ./qiskit-aer/_skbuild/linux-x86_64-${python_version:7:3}/cmake-install/qiskit .
        #rsync -av ./qiskit-aer/_skbuild/linux-x86_64-3.7/cmake-build/qiskit .
    fi 
done

# Clean up and remove the repos
for package in ${QISKIT_PACKAGES[*]}
do
    rm -rf ./$package
done

# Hand over ownership to user
#for package in ${QISKIT_PACKAGES[*]}
#do
#    chown -R $USER ./qiskit 
#done

echo "Qiskit installation is complete!"
