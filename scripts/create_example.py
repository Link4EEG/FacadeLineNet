"""Create an original schematic example, independent of the research dataset."""
from pathlib import Path
from PIL import Image,ImageDraw
def main():
    out=Path(__file__).resolve().parents[1]/'examples';out.mkdir(exist_ok=True)
    im=Image.new('RGB',(512,512),'white');d=ImageDraw.Draw(im)
    d.polygon([(65,160),(256,55),(447,160)],outline='black',width=3)
    d.rectangle((82,160,430,440),outline='black',width=3)
    d.line((75,310,438,310),fill='black',width=3)
    for x in [112,220,328]:
        d.rectangle((x,195,x+68,280),outline='black',width=3)
        d.line((x+34,195,x+34,280),fill='black',width=2)
    for x in [112,328]:d.rectangle((x,337,x+68,408),outline='black',width=3)
    d.rectangle((225,337,287,440),outline='black',width=3)
    d.rectangle((70,440,442,452),outline='black',width=3)
    im.save(out/'schematic_facade.png')
if __name__=='__main__':main()
