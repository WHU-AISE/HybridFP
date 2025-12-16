const sharp = require('sharp');
const fs = require('fs');
const path = require('path');
const https = require('follow-redirects').https;

const filename = "tesla.jpg";
const url = "https://github.com/spcl/serverless-benchmarks-data/blob/6a17a460f289e166abb47ea6298fb939e80e8beb/400.inference/411.image-recognition/fake-resnet/800px-20180630_Tesla_Model_S_70D_2015_midnight_blue_left_front.jpg?raw=true";
const local_path = "./";

function downloadAndProcessImage(url, outputPath, width, height) {
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(outputPath);
        const sharp_resizer = sharp().resize(width, height).png();

        https.get(url, (res) => {
            if (res.statusCode !== 200) {
                reject(new Error(`Download failed with status: ${res.statusCode}`));
                return;
            }

            res.pipe(sharp_resizer).pipe(file);

            file.on('finish', () => {
                file.close();
                resolve(true);
            });

            file.on('error', reject);
            sharp_resizer.on('error', reject);
            res.on('error', reject);
        }).on('error', reject);
    });
}

function checkFileExists(filePath) {
    return new Promise((resolve) => {
        fs.stat(filePath, (err) => {
            resolve(!err);
        });
    });
}

module.exports = async function(event, context = null) {
    let width = event.width || 1000;
    let height = event.height || 1000;

    try {
        // Download and process image
        await downloadAndProcessImage(url, local_path + filename, width, height);
        
        // Check if file was created successfully
        const fileExists = await checkFileExists(local_path + filename);
        
        return {
            "result": fileExists,
            "message": fileExists ? "Image processed successfully" : "Image processing failed",
            "file": filename,
            "dimensions": `${width}x${height}`
        };
    } catch (error) {
        return {
            "error": error.message,
            "success": false
        };
    }
}