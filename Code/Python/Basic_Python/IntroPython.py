# Getting started:

a = 1
b = 2
c = "goober"
c
a+b

s = "Hello World"
print(s)
print("Hello World!")

s = "Today's date is {month}/{day}/{year}".format(month = 10, day = 22, year = 2012)
print(s)

a = 3
b = 4.8878
s = format("sample %d: mass= %0.3fg" % (a, b))
print(s)
print("sample %d: mass= %0.3fg" % (a, b))

# User-defined functions:

def f1(a):
  if a == 0:
    print("hi")
    return(0)
  elif a < 0:
    print("stop")
    return(1)
  else:
    return(5)

f1(1)
f1(-1)

# Line continuation
1+1+1+1+1+1+1+ \
1+1+1+1+1+1+1+1

# Logic and control flow

1 and 2
1 == 1
1 == 1 and 2 == 0
1 > 1 or 2 <= 5
not True
True and not False

if True:
  print("yes")
else:
  print("no")
 
a = 1
if a == 2:
  print("two")
elif a < -1000:
  print("a is small")
elif a > 100 and not a % 2:
  print("a is big and even")
else:
  print("a is none of the above.")

# Strings

a = "Hello World"
b = 'Python is groovy'
c = """Computer says 'No'"""

c = """Computer says 'no'
because another computer
says 'yes'"""

a[0]
a[:5]
a[6:]
a[3:8]

z = "90"
z

int(z)
float(z)
str(90.25)

"123" + "abc"
"123" + str(123.45)

a = 1
b = "2"
str(a) + b

s = "Hello world!"
len(s)
s = "5, 4, 2, 9, 8, 7, 28"
s.count(",")

s.find("9, ")
s[9:12]

"abc123".isalpha()
"abc123".isalnum()

s.split(",")
", ".join(["ready", "set", "go"])
"ready\n set\n go".splitlines()
"ready set go".splitlines()

s = [1, 2, "Five!", ["Three, sir!", "Three!"]]
len(s)
s[0:1]
s[2]
s[2][1]
s[3]
s[3][1]
s[3][1][1]

s.append("new element")
s

l = ["a", "b", "c"]
l.append("d")
l.append("c")
l.remove("a")
l
l.remove("c")
l
l.remove("c")
l

# Tuples

a = ()
b = (3,)
c = (3,4,"thousand")
len(c)

number1, number2, word = c
number1
number2
word

keys =["name", "status", "ID"]
values = ["Joe", "approved", 23425]
z = zip(keys, values)
z

s = "5, 4, 2, 9, 8, 7, 28"

# Dictionaries

stock = {
  "name" : "GOOG",
  "shares" : 100,
  "price" : 490.10 }

stock
stock["name"]
stock["date"] = "today"
stock

keys = ["name", "status", "ID"]
values = ["Joe", "approved", 23425]
zip(keys, values)
d = dict(zip(keys, values))
d

a = "Hello World"
for c in a:
  print c

b = ["Dave","Mark","Ann","Phil"] 
for name in b:
  print name

c = { 'GOOG' : 490.10, 'IBM' : 91.50, 'AAPL' : 123.15 } 
for key in c:
  print key, c[key]

for n in [0, 1,2,3,4,5,6,7,8,9]:
  print("2 to the %d power is %d" % (n, 2**n))

for n in range(9):
  print("2 to the %d power is %d" % (n, 2**n))

# range() and xrange()

range(5)
range(1,8)
range(0, 14, 3)
range(8, 1, -1)

x = 0
for n in xrange(99999):
  x = x + 1
print(x)

# Generators

def countdown(n):
  print "Counting down!" 
  while n > 0:
    yield n # Generate a value (n) 
    n -= 1

c = countdown(5)
c.next()
c.next()
c.next()

nums = [1, 2, 3, 4, 5]
squares = [n * n for n in nums]
squares

a = [-3,5,2,-10,7,8]
b = 'abc'
[2*s for s in a]
[s for s in a if s >= 0]

[(x,y) for x in a for y in b if x > 0 ]
[(1,2), (3,4), (5,6)]

filter(lambda x: x > 3, [0, 1, 2, 3, 4, 5])
l = range(3)
map(str, l)
map(lambda x: x*x, l)

reduce(lambda x, y: x+y, range(1, 11)) # sum the numbers 1 through 10

# File I/O

import random

f = open("data.txt", "w")
f.write("x y\n")
for i in xrange(10):
  f.write("%0.3f %0.3f\n" % (random.random(), random.random()))

f = open("data.txt")
header = f.readline()
data = f.readlines()

header
data

header = header.replace("\n","")
header

d = [d.replace("\n","") for d in data]
d

data = [d.split(" ") for d in data]
data

data = [map(float, d) for d in data]
data


# modules
from math import *
sqrt(10)  

# NumPy
from numpy  import *
a = arange(15)
a

a = a.reshape(3,5)
a

a.transpose()
a.shape
a.size
type(a)

zeros((3,4))
ones((2,3,4), dtype=int16) # dtype can also be specified
empty((2,3))

b = array([[1.5,2,3], [4,5,6]])
b

sum(b)

a = array( [20,30,40,50] )
b = arange( 4 )
b

c = a - b
c

b**2
10 * sin(a)
a < 35

A = array( [[1,1], [0,1]] )
B = array( [[2,0], [3,4]] )
A*B # elementwise product
dot(A, B)

a = array([[ 0,  1,  2,  3,  4], [ 5,  6,  7,  8,  9],[10, 11, 12, 13, 14]])

a
a[0]
a[1]
a[0:2]
a[0,0]
a[1,2]
a[0:2, 0:2]
a[:,:]
a[:, 0]
a[:, 0:1]

for row in a:
  print row
  
for index in xrange(a.shape[1]):
  print a[:, index]

for elt in a.flat:
  print elt,

a = floor(10*random.random((2,2)))
a

b = floor(10*random.random((2,2)))
b

# shallow copying
c = a.view()
c == a
c is a
a[0,0] = 100
a
c

a
b = a
a[0, 0] = 0
b

# deep copying
a
b = a.copy()
b[0,0] = 1000
a
b

# Logical arrays
a = arange(12).reshape(3,4)
b = a > 4
b
a[b]
a[b] = 0
a

# simple linear algebra
from numpy import *
from numpy.linalg import *

a = array([[1.0, 2.0], [3.0, 4.0]])
print a

a.transpose()
inv(A)
u = eye(2)
u

j = array([[0.0, -1.0], [1.0, 0.0]])
dot(j, j)

trace(u)

y = array([[5.], [7.]])
solve(a, y)

eig(j)

# Matrices

A = matrix('1.0 2.0; 3.0 4.0')
A

type(A)

A.T # transpose
X = matrix('5.0 7.0')
Y = X.T
Y

print A*Y  # matrix multiplication FOR MATRICES, elementwise multiplication FOR ARRAYS
print A.I  # inverse

solve(A, Y)  # solving linear equation

A = arange(12).reshape(3,4)
M =  mat(A.copy())

print A[:,1]
print M[:,1]