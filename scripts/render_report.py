"""Build a review PDF from the editable English Markdown report."""
import argparse
from pathlib import Path
from xml.sax.saxutils import escape
import re
import matplotlib
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image

ROOT=Path(__file__).resolve().parents[1]
FONTS=Path(matplotlib.get_data_path())/'fonts/ttf'
pdfmetrics.registerFont(TTFont('Text',str(FONTS/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('Strong',str(FONTS/'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFontFamily('Text',normal='Text',bold='Strong',italic='Text',boldItalic='Strong')
BODY=ParagraphStyle('body',fontName='Text',fontSize=9.2,leading=13.4,spaceAfter=7)
SMALL=ParagraphStyle('small',parent=BODY,fontSize=7.5,leading=10.2)
TITLE=ParagraphStyle('title',parent=BODY,fontName='Strong',fontSize=21,leading=27,spaceAfter=14,keepWithNext=True)
H2=ParagraphStyle('h2',parent=BODY,fontName='Strong',fontSize=15,leading=20,spaceBefore=10,spaceAfter=8,keepWithNext=True)
H3=ParagraphStyle('h3',parent=BODY,fontName='Strong',fontSize=11,leading=15,spaceBefore=5,spaceAfter=5,keepWithNext=True)

def paragraph(text,style=BODY):
    text=escape(text)
    text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
    text=text.replace(chr(96),'')
    text=re.sub(r'\[(.+?)\]\((.+?)\)',r'<u>\1</u> (\2)',text)
    return Paragraph(text,style)

def build(source,target):
    lines=source.read_text(encoding='utf-8').splitlines();story=[];i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('# '):story.append(paragraph(line[2:],TITLE));i+=1;continue
        if line.startswith('## '):story.append(Spacer(1,8));story.append(paragraph(line[3:],H2));i+=1;continue
        if line.startswith('### '):story.append(paragraph(line[4:],H3));i+=1;continue
        if line.startswith('> '):story.append(paragraph(line[2:]));i+=1;continue
        if line.startswith('!['):
            match=re.match(r'!\[(.*?)\]\((.*?)\)',line)
            if match:
                image_path=(ROOT/match.group(2)).resolve()
                with PILImage.open(image_path) as im:w,h=im.size
                width=485;height=width*h/w
                if height>330:height=330;width=height*w/h
                story.extend([Image(str(image_path),width=width,height=height),Spacer(1,4)])
                if match.group(1):story.append(paragraph(match.group(1),SMALL))
            i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                rows.append([cell.strip() for cell in lines[i].strip().strip('|').split('|')]);i+=1
            rows=[row for row in rows if not all(re.fullmatch(r':?-{3,}:?',cell) for cell in row)]
            columns=max(map(len,rows));rows=[row+['']*(columns-len(row)) for row in rows]
            table=Table([[paragraph(cell,SMALL) for cell in row] for row in rows],colWidths=[485/columns]*columns,repeatRows=1)
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EFF4')),('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#205375')),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#BBC8D0')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            story.extend([table,Spacer(1,8)]);continue
        if line.startswith('- '):story.append(paragraph('• '+line[2:]));i+=1;continue
        if re.match(r'^\d+\. ',line):story.append(paragraph(line));i+=1;continue
        parts=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('#','|','![','> ','- ')) and not re.match(r'^\d+\. ',lines[i].strip()):
            parts.append(lines[i].strip());i+=1
        story.append(paragraph(' '.join(parts)))
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#C6D5DF'));canvas.line(50,43,545,43);canvas.setFont('Text',8);canvas.setFillColor(colors.HexColor('#5B7081'));canvas.drawString(50,29,'DSA5205 · editable review copy');canvas.drawRightString(545,29,str(doc.page))
    SimpleDocTemplate(str(target),pagesize=A4,leftMargin=55,rightMargin=55,topMargin=48,bottomMargin=55,title='Editable report draft').build(story,onFirstPage=footer,onLaterPages=footer)
    print(f'Created review PDF: {target}')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',nargs='?',type=Path,default=ROOT/'report.md');parser.add_argument('--output',type=Path,default=ROOT/'outputs/report_preview.pdf');args=parser.parse_args()
    build(args.source.resolve(),args.output.resolve())
