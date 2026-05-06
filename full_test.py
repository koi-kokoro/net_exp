import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageDraw, ImageFont
from ocr_engine import init_ocr_engine, recognize_plate
from plate_detector import detect_plate, cleanup_plate_image

init_ocr_engine()

# Create test image simulating phone photo
img = Image.new('RGB', (1200, 800), color='#8B9DAF')
draw = ImageDraw.Draw(img)
# Car body
draw.rectangle([100, 200, 1100, 750], fill='#E0E0E0', outline='#999')
# Blue plate
draw.rectangle([500, 680, 780, 740], fill='#003399')
draw.rectangle([502, 682, 778, 738], outline='white', width=2)
try:
    font = ImageFont.truetype('c:/Windows/Fonts/simhei.ttf', 32)
except:
    font = ImageFont.load_default()
draw.text((520, 688), 'LuB325DE', fill='white', font=font)
img.save('_full.jpg')
print("Image saved: _full.jpg")

# Step 1: Plate detection
print("\nStep 1: Plate detection...")
detect = detect_plate('_full.jpg')
print("  success=%s method=%s bbox=%s" % (detect['success'], detect['method'], detect['bbox']))

if detect['success']:
    crop_path = detect['plate_image_path']
    print("  crop saved to:", crop_path)
    
    # Step 2: OCR
    print("\nStep 2: OCR on cropped plate...")
    ocr = recognize_plate(crop_path)
    print("  success=%s plate='%s' conf=%.4f" % (ocr['success'], ocr.get('plate','?'), ocr.get('confidence',0)))
    print("  msg: %s" % ocr['msg'])
    cleanup_plate_image(crop_path)
else:
    print("  detection FAILED: %s" % detect['msg'])

# Step 3: Full image OCR for comparison
print("\nStep 3: OCR on full image (no detection)...")
ocr_full = recognize_plate('_full.jpg')
print("  success=%s plate='%s' conf=%.4f" % (ocr_full['success'], ocr_full.get('plate','?'), ocr_full.get('confidence',0)))
print("  msg: %s" % ocr_full['msg'])

os.remove('_full.jpg')
print("\nDONE")
