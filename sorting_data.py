import os
past=""
name="collie"
os.mkdir("F:/Dataset/data/"+name+"_wa")
with open("C:/Users/1/Downloads/"+ name+ "_poses_2.csv") as f:
    for i in f:
        data=i.split(";")
        data[0]=data[0][1:len(data[0])-1]
        if(data[0]!=past):
            os.rename("F:/Dataset/data/"+name+"/"+data[0], "F:/Dataset/data/"+name+"_wa/"+data[0])
        print(data[0])
        past=data[0]
        print(i)