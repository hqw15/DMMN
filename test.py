aa = [1,2,3,4,5,6,7]

def a(x):
    return x > 3

def b(x):
    return x <= 3

def printl(aa, fun):
    for x in aa:
        if fun(x):
            print (x)


printl(aa, a)
print ('==')
printl(aa, b)