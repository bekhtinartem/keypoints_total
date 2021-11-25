f=open("F:/key_points/train_data.csv")
name=""
kk=[]
count=0
for i in f:
    count+=1
    k=i.split(";")
    if(len(k)!=119):
        print(len(k))
    if(k[0]==name):
        if(kk[len(kk)-1][0]==name):
            print(name)
            kk.pop(len(kk)-1)
    else:
        name=k[0]
        kk.append(k)
    #print(kk[len(kk)-1][0])
f.close()

f=open("F:/key_points/test_data.csv")
name=""

for i in f:
    count+=1
    k=i.split(";")
    if(len(k)!=119):
        print(len(k))
    if(k[0]==name):
        if(kk[len(kk)-1][0]==name):
            print(name)
            kk.pop(len(kk)-1)
    else:
        name=k[0]
        kk.append(k)
    #print(kk[len(kk)-1][0])
f.close()

for i in range(len(kk)):
        for j in range(i+1, len(kk)):

            if(kk[i][0]==kk[j][0]):
                kk.pop(j)
                kk.pop(i)
                i-=1
                break
print(len(kk))
print(count)