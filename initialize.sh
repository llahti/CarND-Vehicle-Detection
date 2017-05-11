#!/bin/bash

DATAPATH="./data"

if [ -d "$DATAPATH" ]; then
    echo "Directory $DATAPATH exists, deleting it."
    rm -rf "$DATAPATH"
fi
echo "Create empty directory $DATAPATH"
mkdir $DATAPATH
cd $DATAPATH

echo "Download training data..."
wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip -q --show-progress
unzip -q vehicles.zip
rm vehicles.zip
wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip -q --show-progress
unzip -q non-vehicles.zip
rm non-vehicles.zip
echo ""
echo "Done!"
