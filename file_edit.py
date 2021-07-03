import time
import os
def write_file(a,b):
    file = open("ref_file.txt","rt")
    fout = open("test/test.ino","wt")
    checkWords = ("V1","V2")
    repWords = (str(a),str(b))
    for line in file:
        for check, rep in zip(checkWords, repWords):
            line = line.replace(check, rep)
        fout.write(line)
    file.close()
    fout.close()
    # for line in file:
    #     fout.write(line.replace("V1",str(a)))

    # for line in fout: 
    #     fout.write(line.replace("V2",str(b)))

def term_command():
    os.system("sudo ./upload_sketch.sh")

V1 = range(1000,1400,50)
V2 = range(1500,1900,50)
V1=[1500,1800]
V2=[1600,2000]
for i in range(2):
    write_file(V1[i],V2[i])
    term_command()
    time.sleep(8)