import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageDraw, ImageFont
from ocr_engine import init_ocr_engine, recognize_plate, _do_recognize, _get_reader, _preprocess_image

init_ocr_engine()
reader = _get_reader()

# Create big test image
img = Image.new('RGB', (1200, 800), color='#8B9DAF')
draw = ImageDraw.Draw(img)
draw.rectangle([500, 680, 780, 740], fill='#003399')
try:
    font = ImageFont.truetype('c:/Windows/Fonts/simhei.ttf', 32)
except:
    font = ImageFont.load_default()
draw.text((520, 688), 'LuB325DE', fill='white', font=font)
img.save('_big.jpg')

# Test 1: Direct _do_recognize call
print("Test 1: Direct _do_recognize...")
pp = _preprocess_image('_big.jpg')
print("  pp:", pp)
try:
    result = _do_recognize(reader, pp)
    print("  result:", result)
except Exception as e:
    import traceback
    traceback.print_exc()

# Test 2: Through recognize_plate
print("\nTest 2: Through recognize_plate...")
try:
    result = recognize_plate('_big.jpg')
    print("  result:", result)
except Exception as e:
    import traceback
    traceback.print_exc()

os.remove('_big.jpg')
print("\nDONE")
