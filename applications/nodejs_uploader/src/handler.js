const fs = require('fs');
const https = require('https');

const filename = "hpx.zip";
const url = "https://mirrors.huaweicloud.com/nginx/nginx-1.24.0.zip";
const local_path = "./";

function downloadToFile(url, filePath) {
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(filePath);
        
        https.get(url, (res) => {
            if (res.statusCode !== 200) {
                reject(new Error(`Request failed with status code: ${res.statusCode}`));
                return;
            }
            
            res.pipe(file);
            
            file.on('finish', () => {
                file.close();
                resolve();
            });
            
            file.on('error', reject);
            res.on('error', reject);
        }).on('error', reject);
    });
}

module.exports = async function(event, context = null) {
    let couch_link = "http://whisk_admin:some_passw0rd@172.17.0.1:5984";
    let db_name = "ul";

    // Connect to couchdb
    const nano = require('nano')(couch_link);
    try {
        nano.use(db_name);
    } catch (e) {
        await nano.db.create(db_name);
        const database = nano.use(db_name);
        await database.insert({"success": true}, 'file');
    }
    const database = nano.use(db_name);
    var doc = await database.get('file');

    try {
        // Download file
        await downloadToFile(url, local_path + filename);
        
        // Read file and upload to CouchDB
        var data = fs.readFileSync(local_path + filename);
        await database.attachment.insert(doc._id, filename, data, "application/zip", {'rev': doc._rev});
        
        return {"result": doc, "success": true};
    } catch (error) {
        console.error("Error:", error);
        return {"error": error.message, "success": false};
    }
}