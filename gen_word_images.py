# Functions generating images of words for training the CORnet model. 
from PIL import Image, ImageDraw, ImageFont, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
import os, random, gc
import numpy as np
from pathlib import Path

base_dir = Path('/project/3011213.01/Origins-of-VWFA/')

def CreateWordSet(path_out=base_dir / 'wordsets', num_train=100, num_val=50, num_test_acts=100):
    #define words, sizes, fonts
    words = [line.strip() for line in open(base_dir / 'words_dutch_cornet.txt', 'r')]

    sizes = [40, 50, 60, 70, 80]
    fonts = {'train': ['arial', 'times', 'lcd'], 
             'val': ['comic', 'cour', 'lcd'],
             'test_acts': ['lcd']}
    xshift = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
    yshift = [-30, -15, 0, 15, 30]
    min_col_diff = 40
    
    #create train and val folders 
    for m in ['train', 'val', 'test_acts']:
        for f in words:
            target_path = path_out / m / f
            target_path.mkdir(parents=True, exist_ok=True)
    
    #for each word, create num_train + num_val exemplars, then split randomly into train and val.
    for w in tqdm(words):
        gc.collect()
        print (w,)
        for n in range(num_train + num_val + num_test_acts):
            if n < num_train:
                stage = 'train'
            elif n < num_train + num_val:
                stage = 'val'
            else:
                stage = 'test_acts'

            path = path_out / stage / w
            
            f = random.choice(fonts[stage])
            s = random.choice(sizes)
            u = random.choice([0,1])
            x = random.choice(xshift)
            y = random.choice(yshift)
            
            # Determine background and foreground (grey) colors such that the difference in brightness is at least 30. 
            # The foreground can be both lighter and darker than the background.
            bg = random.randint(0, 255)
            diff = random.randint(0, 255 - min_col_diff * 2)
            if diff < bg - min_col_diff:
                fg = bg - diff - min_col_diff
            else:    
                fg = bg + diff + min_col_diff

            if f == 'lcd':
                img = gen_wordimg_lcd(text=w, W=500, H=500, size=s, xshift=x, yshift=y, bg=bg, fg=fg)
            else:
                img = gen_wordimg_ttf(text=w, W=500, H=500, fontname=f, size=s, xshift=x, yshift=y, upper=u, bg=bg, fg=fg)

            if path != '':
                img.save(path / f'{n}.jpg')

def gen_wordimg_ttf(text='text', fontname='Arial', W=500, H=500, size=24, xshift=0, yshift=0, upper=0, bg=255, fg=0):
    if upper:
        text = text.upper()
    img = Image.new("RGB", (W,H), color = (bg, bg, bg))
    #fnt = ImageFont.truetype('/Library/Fonts/'+fontname+'.ttf', size) #size in pixels
    fnt = ImageFont.truetype(f'fonts/{fontname}.ttf', size)
    draw = ImageDraw.Draw(img)

    # w, h = fnt.getsize(text)
    bbox = fnt.getbbox(text)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    draw.text((xshift + (W-w)/2, yshift + (H-h)/2), text, font=fnt, fill=(fg, fg, fg))

    return img


def gen_wordimg_lcd(text='text', W=500, H=500, size=24, xshift=0, yshift=0, bg=255, fg=0):
    text = text.upper()
    slots = ['0_0_1', '0_0_2', '0_0_3', 
            '1_0_1', '1_0_2', '1_0_3', '1_0_4',
            '2_0_1', '2_0_4', 
            '0_1_1', '0_1_2', '0_1_3', 
            '1_1_1', '1_1_2', '1_1_3', '1_1_4', 
            '2_1_1', '2_1_4', 
            '0_2_2', '1_2_2'
            ]

    letters = {}
    letters['A'] = ['1_0_4', '1_0_1', '1_0_2', '0_1_3', '1_1_1', '1_2_2']
    letters['B'] = ['0_0_1', '0_0_2', '0_1_1', '0_2_2', '1_0_1', '1_0_2', '1_1_1', '1_2_2', '2_0_1', '2_1_1']
    letters['C'] = ['0_0_1', '0_0_2', '0_1_1', '1_0_2', '2_0_1', '2_1_1']
    letters['D'] = ['0_0_1', '0_0_2', '0_1_3', '1_0_2', '2_0_1', "2_1_4"]
    letters['E'] = ['0_0_1', '0_0_2', '0_1_1', '1_0_1', '1_0_2', '1_1_1', '2_0_1', '2_1_1']
    letters['F'] = ['0_0_1', '0_0_2', '0_1_1', '1_0_1', '1_0_2']
    letters['G'] = ['0_0_1', '0_0_2', '0_1_1', '1_0_2', '1_1_1', '1_2_2', '2_0_1', '2_1_1']
    letters['H'] = ['0_0_2', '0_2_2', '1_0_1', '1_0_2', '1_1_1', '1_2_2']
    letters['I'] = ['0_0_1', '0_1_1', '0_1_2', '1_1_2', '2_0_1', '2_1_1']
    letters['J'] = ['0_1_1', '0_2_2', '1_0_2', '1_2_2', '2_0_1', '2_1_1']
    letters['K'] = ['0_0_2', '1_0_1', '1_0_2', '1_1_3', '1_1_4']
    letters['L'] = ['0_0_2', '1_0_2', '2_0_1', '2_1_1']
    letters['M'] = ['0_0_2', '0_0_3', '0_2_2', '1_0_2', '1_1_4', '1_2_2']
    letters['N'] = ['0_0_2', '0_0_3', '0_2_2', '1_0_2', '1_1_3', '1_2_2']
    letters['O'] = ['0_0_1', '0_0_2', '0_1_1', '0_2_2', '1_0_2', '1_2_2', '2_0_1', '2_1_1']
    letters['P'] = ['0_0_1', '0_0_2', '0_1_1', '0_2_2', '1_0_1', '1_0_2', '1_1_1']
    letters['Q'] = ['0_0_1', '0_0_2', '0_1_1', '0_2_2', '1_0_2', '1_1_3', '1_2_2', '2_0_1', '2_1_1']
    letters['R'] = ['0_0_1', '0_0_2', '0_1_1', '0_2_2', '1_0_1', '1_0_2', '1_1_1', '1_1_3']
    letters['S'] = ['0_0_1', '0_0_2', '0_1_1', '1_0_1', '1_1_1', '1_2_2', '2_0_1', '2_1_1']
    letters['T'] = ['0_0_1', '0_1_1', '0_1_2', '1_1_2']
    letters['U'] = ['0_0_2', '0_2_2', '1_0_2', '1_2_2', '2_0_1', '2_1_1']
    letters['V'] = ['0_0_2', '0_2_2', '1_0_3', '2_1_4']
    letters['W'] = ['0_0_2', '0_2_2', '1_0_2', '1_2_2', '2_0_4', '1_1_3']
    letters['X'] = ['0_0_3', '2_0_4', '1_1_3', '1_1_4']
    letters['Y'] = ['0_0_3', '1_1_2', '1_1_4']
    letters['Z'] = ['0_0_1', '0_1_1', '1_1_4', '2_0_4', '2_0_1', '2_1_1']
    letters[' '] = []


    # Arial/Times sizes in font size vs pixels
    # 53-58 = 80px
    # 27-29 = 40px


    s = 0.5 #w/h ratio
    width_prop = 0.16
    padding_prop = 0.25

    linelen = 100 # Super-sampled version; size * 0.3375
    width = int(round(linelen*width_prop))
    linelen = int(round(linelen))

    def draw_line(draw, slot, fill=255, offset=0):
        slot = [int(s) for s in slot.split('_')]
        xy1 = xy(slot[0], slot[1])
        xy1 = (xy1[0]+offset, xy1[1])
        dir = slot[2]
        fill = int(round(fill))
        fill = (fill, fill, fill)
        
        if dir == 1:  # horizontal, right
            hw = width/2
            xy2 = (xy1[0]+linelen*s, xy1[1])
            draw.polygon(((xy1[0], xy1[1]), (xy1[0]+hw, xy1[1]-hw),
                        (xy2[0]-hw, xy2[1]-hw), (xy2[0], xy2[1]),
                        (xy2[0]-hw, xy2[1]+hw), (xy1[0]+hw, xy2[1]+hw)), fill=fill)
        elif dir == 2: #vertical, down
            hw = width/2
            xy2 = (xy1[0], xy1[1]+linelen)
            draw.polygon(((xy1[0], xy1[1]), (xy1[0]+hw, xy1[1]+hw),
                        (xy2[0]+hw, xy2[1]-hw), (xy2[0], xy2[1]),
                        (xy2[0]-hw, xy2[1]-hw), (xy1[0]-hw, xy1[1]+hw)), fill=fill)
        elif dir == 3: #oblique, down-right
            xy1 = (xy1[0] + width/2, xy1[1] + width/2)
            xy2 = (xy1[0] + linelen*s-width, xy1[1]+linelen-width)
            diaghw_v = width * 1.1 #na oko, musialbym to przeliczyc duzo staranniej
            diaghw_h = diaghw_v*s
            draw.polygon(((xy1[0], xy1[1]), (xy1[0]+diaghw_h, xy1[1]), 
                        (xy2[0], xy2[1]-diaghw_v), (xy2[0], xy2[1]), 
                        (xy2[0]-diaghw_h, xy2[1]), (xy1[0], xy1[1]+diaghw_v)), fill=fill)
        elif dir == 4: #oblique, up-right
            xy1 = (xy1[0] + width/2, xy1[1] - width/2)
            xy2 = (xy1[0] + linelen*s-width, xy1[1]-(linelen-width))
            diaghw_v = width * 1.1
            diaghw_h = diaghw_v*s
            draw.polygon(((xy1[0], xy1[1]), (xy1[0]+diaghw_h, xy1[1]), 
                        (xy2[0], xy2[1]+diaghw_v), (xy2[0], xy2[1]), 
                        (xy2[0]-diaghw_h, xy2[1]), (xy1[0], xy1[1]-diaghw_v)), fill=fill)

    def xy(row, col):
        x = width/2 + col*linelen*s
        y = width/2 + row*linelen
        return (x, y)

    def draw_letter(draw, grid, pos):
        offset = pos * (linelen*s*2 + width + linelen*s*padding_prop)
        
        slots = sorted(grid.keys(), key=lambda item: grid[item])
        for slot in slots:
            draw_line(draw, slot, fill=grid[slot], offset=offset)

    def make_empty_grid():
        ret = {}
        for slot in slots:
            ret[slot] = bg
        return ret

    n = len(text)
    image_txt = Image.new("RGB", (int(round(n*(linelen*s*2 + width) + (n-1)*linelen*s*padding_prop)),
                                  linelen*2+width), (bg, bg, bg))
    draw = ImageDraw.Draw(image_txt)
    for lx in range(n):
        letter = text[lx]
        grid = make_empty_grid()
        for slot in letters[letter]:
            grid[slot] = fg # grid[slot]*(1-transparency) + 255*transparency
        draw_letter(draw, grid, lx)

    # Downsampling
    w, h = image_txt.size
    tgt_h = int(round(size * 0.725 * 1.4)) # *1.4 if match to width of 'E' in Arial
    tgt_w = int(round(w * tgt_h / h))
    image_txt_ds = image_txt.resize((tgt_w, tgt_h), Image.Resampling.BILINEAR)

    image_canvas = Image.new("RGB", (W, H), color = (bg, bg, bg))
    w, h = image_txt_ds.size

    image_canvas.paste(image_txt_ds, (round(xshift + (W - w) / 2), round(yshift + (H - h) / 2)))
    return image_canvas



def main():
    CreateWordSet(num_train=1300, num_val=50, num_test_acts=1)

if __name__ == "__main__":
    main()
