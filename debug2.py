import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageDraw, ImageFont
from ocr_engine import init_ocr_engine, _get_reader, _preprocess_image

init_ocr_engine()
reader = _get_reader()

# Create big test image
img = Image.new('RGB', (1200, 800), color='#8B9DAF')
draw = ImageDraw.Draw(img)
draw.rectangle([500, 680, 780, 740], fill='#003399')
draw.rectangle([502, 682, 778, 738], outline='white', width=2)
try:
    font = ImageFont.truetype('c:/Windows/Fonts/simhei.ttf', 32)
except:
    font = ImageFont.load_default()
draw.text((520, 688), 'LuB325DE', fill='white', font=font)
img.save('_big.jpg')

pp = _preprocess_image('_big.jpg')
print("preprocessed:", pp)
print("size:", Image.open(pp).size)

print("\nCalling reader.readtext...")
results = reader.readtext(pp)
print("results type:", type(results))
print("results len:", len(results))
if results:
    item = results[0]
    print("item type:", type(item))
    print("item len:", len(item) if hasattr(item, '__len__') else 'n/a')
    print("item:", item)
else:
    print("results is EMPTY!")

os.remove('_big.jpg')
if pp != '_big.jpg':
    os.remove(pp)
print("\nDONE")
