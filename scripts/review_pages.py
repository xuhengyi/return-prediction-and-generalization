"""Make a local contact sheet of already rendered report pages."""
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw

root=Path('outputs/review');pages=sorted(root.glob('page-*.png'))
sheet=Image.new('RGB',(1200,4*460),'#CAD2D8');draw=ImageDraw.Draw(sheet)
for i,path in enumerate(pages):
    im=Image.open(path).convert('RGB');im.thumbnail((380,420))
    x=(i%3)*400+(400-im.width)//2;y=(i//3)*460+25
    sheet.paste(im,(x,y));draw.text((i%3*400+15,i//3*460+5),f'Page {i+1}',fill='black')
sheet.save(root/'contact_sheet.png')
sheet.save(root/'contact_sheet.jpg',quality=80)
