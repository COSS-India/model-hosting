# Quick Start Guide - Surya OCR Triton

## 🚀 How to Use Your Own Images

### Method 1: Using `test_hindi_ocr.py` (Recommended for Testing)

This script can now accept your own PNG/JPG images!

```bash
# Test with your own image
python3 test_hindi_ocr.py --image your_document.png

# Short form
python3 test_hindi_ocr.py -i receipt.jpg

# Test with auto-generated Hindi text (default)
python3 test_hindi_ocr.py
```

**Features:**
- ✅ Accepts PNG, JPG, JPEG, BMP, TIFF, WebP
- ✅ Shows detailed results with confidence scores
- ✅ Displays bounding boxes and coordinates
- ✅ Detects Hindi (Devanagari) characters
- ✅ Works with any language (90+ supported)

---

### Method 2: Using `ocr_image.py` (Simple & Fast)

For quick OCR without detailed analysis:

```bash
# Basic usage - just get the text
python3 ocr_image.py your_image.png

# Get detailed results with coordinates
python3 ocr_image.py your_image.png --detailed

# Save results to a text file
python3 ocr_image.py your_image.png --output result.txt

# Get JSON output for further processing
python3 ocr_image.py your_image.png --json
```

---

## 📋 Step-by-Step Example

### Example 1: OCR a Receipt

```bash
# 1. Copy your receipt image to the directory
cp ~/Downloads/receipt.jpg .

# 2. Run OCR
python3 test_hindi_ocr.py --image receipt.jpg

# Or use the simpler script
python3 ocr_image.py receipt.jpg
```

### Example 2: OCR a Hindi Document

```bash
# 1. Have your Hindi document ready
# 2. Run OCR with detailed output
python3 test_hindi_ocr.py --image hindi_doc.png

# 3. Save the extracted text
python3 ocr_image.py hindi_doc.png --output hindi_text.txt
```

### Example 3: Batch Process Multiple Images

Create a simple bash script `batch_ocr.sh`:

```bash
#!/bin/bash

# Process all PNG images in current directory
for img in *.png; do
    echo "Processing: $img"
    python3 ocr_image.py "$img" --output "${img%.png}.txt"
done

echo "Done! All text files created."
```

Run it:
```bash
chmod +x batch_ocr.sh
./batch_ocr.sh
```

---

## 🎯 Command Reference

### `test_hindi_ocr.py` - Full Testing Script

```bash
# Show help
python3 test_hindi_ocr.py --help

# Test with your image
python3 test_hindi_ocr.py --image <path>
python3 test_hindi_ocr.py -i <path>

# Use custom server URL
python3 test_hindi_ocr.py --image doc.png --server http://localhost:8400

# Generate Hindi test image (default behavior)
python3 test_hindi_ocr.py
```

**Output includes:**
- Full text extraction
- Line-by-line detailed results
- Confidence scores for each line
- Bounding box coordinates
- Polygon coordinates
- Average confidence
- Hindi character detection status

---

### `ocr_image.py` - Simple OCR Script

```bash
# Show help
python3 ocr_image.py --help

# Basic usage
python3 ocr_image.py <image_path>

# Detailed output
python3 ocr_image.py <image_path> --detailed
python3 ocr_image.py <image_path> -d

# Save to file
python3 ocr_image.py <image_path> --output <file>
python3 ocr_image.py <image_path> -o result.txt

# JSON output
python3 ocr_image.py <image_path> --json
python3 ocr_image.py <image_path> -j

# Custom server
python3 ocr_image.py <image_path> --server http://localhost:8400
```

---

## 📊 Understanding the Output

### Simple Output (default)
```
======================================================================
OCR RESULTS: your_image.png
======================================================================

Your extracted text appears here
Line by line
As it was detected in the image

======================================================================
Detected 3 text lines
Average confidence: 98.50%
======================================================================
```

### Detailed Output (--detailed flag)
```
======================================================================
DETAILED OCR RESULTS: your_image.png
======================================================================

Line 1:
  Text: Your extracted text
  Confidence: 0.9850 (98.50%)
  BBox: [48.0, 42.0, 374.0, 68.0]
  Polygon: [[48.0, 42.0], [374.0, 42.0], [374.0, 68.0], [48.0, 68.0]]

Line 2:
  Text: Second line of text
  Confidence: 0.9920 (99.20%)
  BBox: [48.0, 123.0, 478.0, 162.0]
  Polygon: [[48.0, 123.0], [478.0, 123.0], [478.0, 162.0], [48.0, 162.0]]

======================================================================
Total lines: 2
Average confidence: 0.9885 (98.85%)
======================================================================
```

### JSON Output (--json flag)
```json
{
  "success": true,
  "full_text": "Your extracted text\nSecond line of text",
  "text_lines": [
    {
      "text": "Your extracted text",
      "confidence": 0.985,
      "bbox": [48.0, 42.0, 374.0, 68.0],
      "polygon": [[48.0, 42.0], [374.0, 42.0], [374.0, 68.0], [48.0, 68.0]]
    }
  ],
  "image_bbox": [0, 0, 800, 600]
}
```

---

## 🌍 Supported Languages

Works with **90+ languages** including:

**Indian Languages:**
- Hindi (हिंदी) ✅ Tested - 98.95% accuracy
- Bengali (বাংলা)
- Tamil (தமிழ்)
- Telugu (తెలుగు)
- Marathi (मराठी)
- Gujarati (ગુજરાતી)
- Kannada (ಕನ್ನಡ)
- Malayalam (മലയാളം)
- Punjabi (ਪੰਜਾਬੀ)
- Urdu (اردو)

**Other Languages:**
- English, Spanish, French, German, Italian, Portuguese
- Chinese, Japanese, Korean
- Arabic, Persian, Hebrew
- Thai, Vietnamese, Indonesian
- And many more!

---

## 🖼️ Supported Image Formats

- ✅ PNG
- ✅ JPEG / JPG
- ✅ BMP
- ✅ TIFF
- ✅ WebP

---

## 💡 Tips for Best Results

1. **Image Quality**: Use high-resolution images (300 DPI or higher)
2. **Contrast**: Ensure good contrast between text and background
3. **Orientation**: Make sure image is properly oriented (not rotated)
4. **Lighting**: Avoid shadows and glare
5. **Focus**: Text should be sharp and in focus

---

## 🔧 Server Configuration

**Default Server**: `http://65.1.35.3:8400`

To use a different server:
```bash
python3 test_hindi_ocr.py --image doc.png --server http://localhost:8400
python3 ocr_image.py doc.png --server http://localhost:8400
```

---

## 🐛 Troubleshooting

### Error: "File not found"
```bash
# Make sure the file path is correct
ls -lh your_image.png

# Use absolute path if needed
python3 ocr_image.py /full/path/to/image.png
```

### Error: "Connection refused"
```bash
# Check if server is running
curl http://65.1.35.3:8400/v2/health/ready

# If not running, start it
docker ps | grep surya-ocr-triton
```

### Low Confidence Scores
- Check image quality and resolution
- Ensure proper lighting and contrast
- Verify text is not too small or blurry

---

## 📚 More Information

- **Full Documentation**: `README.md`
- **Usage Examples**: `USAGE_EXAMPLES.md`
- **Hindi Test Results**: `HINDI_OCR_TEST_RESULTS.md`
- **Testing Guide**: `TESTING_GUIDE.md`

---

## 🎓 Real-World Examples

### Example 1: Extract Text from Screenshot
```bash
python3 ocr_image.py screenshot.png --output extracted.txt
cat extracted.txt
```

### Example 2: OCR Multiple Documents
```bash
for doc in document*.png; do
    python3 ocr_image.py "$doc" -o "${doc%.png}.txt"
done
```

### Example 3: Get JSON for API Integration
```bash
python3 ocr_image.py invoice.jpg --json > invoice_data.json
cat invoice_data.json | jq '.full_text'
```

---

**Ready to start? Just run:**

```bash
python3 test_hindi_ocr.py --image your_image.png
```

or

```bash
python3 ocr_image.py your_image.png
```

**That's it! 🚀**

