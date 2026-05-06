import urllib.request, json, os, io
from PIL import Image, ImageDraw, ImageFont

# Create test plate image
img = Image.new('RGB', (500, 120), color='#003399')
d = ImageDraw.Draw(img)
d.rectangle([0, 0, 499, 119], outline='white', width=3)
try:
    f = ImageFont.truetype('c:/Windows/Fonts/simhei.ttf', 50)
except:
    f = ImageFont.load_default()
d.text((20, 30), 'LuB325DE', fill='white', font=f)
img.save('_test.jpg')

# Upload
boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = b''
body += b'--' + boundary.encode() + b'\r\n'
body += b'Content-Disposition: form-data; name="image"; filename="test.jpg"\r\n'
body += b'Content-Type: image/jpeg\r\n\r\n'
with open('_test.jpg', 'rb') as f:
    body += f.read()
body += b'\r\n--' + boundary.encode() + b'--\r\n'

req = urllib.request.Request(
    'http://127.0.0.1:5000/api/upload',
    data=body,
    headers={'Content-Type': 'multipart/form-data; boundary=' + boundary}
)
r = urllib.request.urlopen(req)
resp = json.loads(r.read())
print('plate:', resp.get('plate'))
print('success:', resp.get('code') == 200)
print('conf:', resp.get('confidence'))
print('msg:', resp.get('msg'))

os.remove('_test.jpg')
