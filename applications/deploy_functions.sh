#! /bin/bash

set -e

# Function configurations
memory=256
timeout=300000

echo ""
echo "Deploying functions..."
echo ""

#
# Auto Complete (ac)
#

cd nodejs_auto_complete

# Destroy and prepare build folder.
rm -rf build
mkdir build

# Copy files to build folder.
cp -R src/* build
cd build && npm install
zip -r index.zip *

wsk -i action update ac --kind nodejs:18 --main main --memory 128 --timeout $timeout index.zip

cd ../../

#
# Image Sizing (is)
#

cd nodejs_image_sizing

# Destroy and prepare build folder.
rm -rf build
mkdir build

# Copy files to build folder.
cp -R src/* build
cd build && npm install
zip -r index.zip *

wsk -i action update is --kind nodejs:18 --main main --memory 256 --timeout $timeout index.zip

cd ../../

#
# OCR Image (oi)
#

cd nodejs_ocr_image

# Destroy and prepare build folder.
rm -rf build
mkdir build

# Copy files to build folder.
cp -R src/* build
cd build && npm install
zip -r index.zip *

wsk -i action update oi --kind nodejs:18 --main main --memory 257 --timeout $timeout index.zip

cd ../../

#
# Dynamic Html (dh)
#

cd nodejs_dynamic_html

# Destroy and prepare build folder.
rm -rf build
mkdir build

# Copy files to build folder.
cp -R src/* build
cd build && npm install
zip -r index.zip *

wsk -i action update dh --kind nodejs:18 --main main --memory 129 --timeout $timeout index.zip

cd ../../

#
# Uploader (ul)
#

cd nodejs_uploader

# Destroy and prepare build folder.
rm -rf build
mkdir build

# Copy files to build folder.
cp -R src/* build
cd build && npm install
zip -r index.zip *

# Create couchdb database
curl -X PUT "http://whisk_admin:some_passw0rd@172.17.0.1:5984/ul"
curl -X PUT "http://whisk_admin:some_passw0rd@172.17.0.1:5984/ul/'file'" -d '{"success": true}'

wsk -i action update ul --kind nodejs:18 --main main --memory $memory --timeout $timeout index.zip

cd ../../

#
# Thumbnailer (tn)
#

cd nodejs_thumbnailer

# Destroy and prepare build folder.
rm -rf build
mkdir build

# Copy files to build folder.
cp -R src/* build
cd build && npm install
zip -r index.zip *

wsk -i action update tn --kind nodejs:18 --main main --memory 130 --timeout $timeout index.zip

cd ../../

#
# File Compression (fc)
#

cd python_file_compression

# Destroy and prepare build folder.
rm -rf build
mkdir build

# Copy files to build folder.
cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update fc --kind python:3.10 --main main --memory 128 --timeout $timeout index.zip

cd ../../

# #
# # Video Processing (vp)
# #

# cd python_video_processing

# # Destroy and prepare build folder.
# rm -rf build
# mkdir build

# # Copy files to build folder.
# cp -R src/* build
# cd build
# zip -r index.zip *

# wsk -i action update vp --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip

# cd ../../

# #
# # Image Recognition (ir)
# #

# cd python_image_recognition

# # Destroy and prepare build folder.
# rm -rf build
# mkdir build

# # Copy files to build folder.
# cp -R src/* build
# cd build
# zip -r index.zip *

# wsk -i action update ir --kind python:3.10 --main main --memory $memory --timeout $timeout index.zip

# cd ../../

#
# Sentiment Analysis (sa)
#

cd python_sentiment_analysis

# Destroy and prepare build folder.
rm -rf build
mkdir build

# Copy files to build folder.
cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update sa --kind python3 --main main --memory 129 --timeout $timeout index.zip
# wsk -i action update sa --docker day1zz/python3action:ml-libs --main main --memory $memory --timeout $timeout index.zip
cd ../../

#
# DNA Visualisation (dv)
#

cd python_dna_visualization

# Destroy and prepare build folder.
rm -rf build
mkdir build

# Copy files to build folder.
cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update dv --kind python3 --main main --memory 130 --timeout $timeout index.zip
# wsk -i action update dv --docker day1zz/python3action:ml-libs --main main --memory $memory --timeout $timeout index.zip


cd ../../

#
# Markdown (md)
#

cd python_markdown

# Destroy and prepare build folder.
rm -rf build
mkdir build

# Copy files to build folder.
cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update md --kind python3 --main main --memory 131 --timeout $timeout index.zip
# wsk -i action update md --docker day1zz/python3action:ml-libs --main main --memory $memory --timeout $timeout index.zip

cd ../../

#
# Graph BFS (gb)
#

cd python_graph_bfs

# Destroy and prepare build folder.
rm -rf build
mkdir build

# Copy files to build folder.
cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update gb --kind python3 --main main --memory 132 --timeout $timeout index.zip
# wsk -i action update gb --docker day1zz/python3action:ml-libs --main main --memory $memory --timeout $timeout index.zip

cd ../../

#
# Graph MST (gm)
#

cd python_graph_mst

# Destroy and prepare build folder.
rm -rf build
mkdir build

# Copy files to build folder.
cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update gm --kind python3 --main main --memory 133 --timeout $timeout index.zip
# wsk -i action update gm --docker day1zz/python3action:ml-libs --main main --memory $memory --timeout $timeout index.zip

cd ../../

#
# Graph Pangrank (gp)
#

cd python_graph_pagerank

# Destroy and prepare build folder.
rm -rf build
mkdir build

# Copy files to build folder.
cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update gp --kind python3 --main main --memory 128 --timeout $timeout index.zip
# wsk -i action update gp --docker day1zz/python3action:ml-libs --main main --memory $memory --timeout $timeout index.zip

cd ../../

#
# Data Transform (dt)
#

cd java_data_transform

# Compile jar file
mvn clean verify -f pom.xml

wsk -i action update dt --kind java:8 --memory 128 --timeout $timeout --main openwhisk.Main target/main-1.0-SNAPSHOT.jar

cd ../

#
# Data Load (dl)
#

cd java_data_load

# Compile jar file
mvn clean verify -f pom.xml

wsk -i action update dl --kind java:8 --memory 129 --timeout $timeout --main openwhisk.Main target/main-1.0-SNAPSHOT.jar

cd ../

#
# Data Query (dq)
#

cd java_data_query

# Compile jar file
mvn clean verify -f pom.xml

wsk -i action update dq --kind java:8 --memory 130 --timeout $timeout --main openwhisk.Main target/main-1.0-SNAPSHOT.jar

cd ../

#
# Data Scan (ds)
#

cd java_data_scan

# Compile jar file
mvn clean verify -f pom.xml

wsk -i action update ds --kind java:8 --memory 131 --timeout $timeout --main openwhisk.Main target/main-1.0-SNAPSHOT.jar

cd ../

#
# Data Group (dg)
#

cd java_data_group

# Compile jar file
mvn clean verify -f pom.xml

wsk -i action update dg --kind java:8 --memory 132 --timeout $timeout --main openwhisk.Main target/main-1.0-SNAPSHOT.jar

cd ../

#
# End Experiment
#

cd python_end_experiment

# Destroy and prepare build folder.
rm -rf build
mkdir build

# Copy files to build folder.
cp -R src/* build
cd build
zip -r index.zip *

wsk -i action update EndExperiment --kind python:3.10 --timeout $timeout --main main --memory $memory index.zip

cd ../../

echo ""
echo "Finish deployment!"
echo ""
