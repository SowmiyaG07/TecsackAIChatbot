import pymupdf
import pytesseract
from PIL import Image
from pytesseract import Output


file_path = "data/hr/Holidaylist2026.pdf"

doc = pymupdf.open(file_path)

page = doc[0]

pix = page.get_pixmap(
    matrix=pymupdf.Matrix(3, 3),
    alpha=False,
)

image = Image.frombytes(
    "RGB",
    [pix.width, pix.height],
    pix.samples,
)


data = pytesseract.image_to_data(
    image,
    config="--psm 4",
    output_type=Output.DICT,
)


print()
print("=" * 100)
print("OCR WORD COORDINATES")
print("=" * 100)


for i in range(len(data["text"])):

    text = data["text"][i].strip()

    if not text:
        continue

    confidence = data["conf"][i]

    x = data["left"][i]
    y = data["top"][i]
    width = data["width"][i]
    height = data["height"][i]

    print(
        f"text={text!r:35} "
        f"x={x:5} "
        f"y={y:5} "
        f"w={width:4} "
        f"h={height:4} "
        f"conf={confidence}"
    )


doc.close()