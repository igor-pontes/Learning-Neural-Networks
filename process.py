import os

def processfile(_file):
    images = []
    labels = []
    content = []
    file = open(os.path.dirname(os.path.realpath(__file__))+"/dataset/"+_file, 'rb')
    try: content = file.read()
    finally: file.close()
    magic_number = int.from_bytes(content[0:4], "big")
    total = int.from_bytes(content[4:8], "big")
    c = 0
    if magic_number == 2051:
        rows = int.from_bytes(content[8:12], "big")
        columns = int.from_bytes(content[12:16], "big")
        i = 16
        while c < total:
            try:
                image = []
                image = content[i:i+(rows*columns)]
                images.append(image)
                i = i+(rows*columns)
            except IndexError:
                print("Error")
                break
            c += 1
        return images
    if magic_number == 2049:
        i = 8
        while c < total:
            try:
                labels.append(content[i])
                i += 1
            except IndexError:
                print("Error")
                break
            c += 1
        return labels

