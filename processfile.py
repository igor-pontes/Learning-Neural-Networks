import os

def processfile(imagefile):
    images = []
    labels = []
    file = open(os.path.dirname(os.path.realpath(__file__))+"/dataset/"+imagefile, 'rb')
    with file as f:
        content = f.readlines()
    print(len(content))
    header = []
    header.append(content[0][0:4])
    header.append(content[0][4:8])
    header.append(content[0][8:12])
    header.append(content[0][12:16])
    magic_number = int.from_bytes(header[0], "big")
    total = int.from_bytes(header[1], "big")
    print(total)
    if magic_number == 2051:
        rows = int.from_bytes(header[2], "big")
        columns = int.from_bytes(header[3], "big")
        image = []
        i = 16
        j = 0
        while(1):
            try:
                if j == columns:
                    image = []
                    j = 0
                for i in range(i, i + rows):
                    image.append(content[i])
                j += 1
                images.append(image)
                i = i + rows + 1
            except IndexError:
                break
        file.close()
        return images
    if magic_number == 2049:
        i = 8
        while(1):
            try:
                labels.append(content[0][i])
                i += 1
            except IndexError:
                break
        file.close()
        return labels
#data = processfile("train-images-idx3-ubyte")
data = processfile("train-labels-idx1-ubyte")
print(len(data))
