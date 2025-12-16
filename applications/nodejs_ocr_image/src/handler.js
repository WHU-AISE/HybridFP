const { createWorker } = require('tesseract.js');

const filename = "exodia.png";
const local_path = "./";

async function ocr(imagePath) {
    const worker = await createWorker('eng');  // 使用英文语言包
    
    try {
        const { data: { text } } = await worker.recognize(imagePath);
        return {
            statusCode: 200,
            body: text
        };
    } catch (error) {
        return {
            statusCode: 500,
            body: `OCR Error: ${error.message}`
        };
    } finally {
        await worker.terminate();
    }
}

module.exports = async function(event, context = null) {
    try {
        const result = await ocr(local_path + filename);
        return result;
    } catch (error) {
        return {
            statusCode: 500,
            body: `Processing failed: ${error.message}`
        };
    }
}