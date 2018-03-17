import math 
import sys 
from booksite import stddraw, stdarray

n = int(sys.argv[1])

x = stdarray.create1D(n + 1, 0.0)
y = stdarray.create1D(n + 1, 0.0)
for i in range(n+1):
    x[i] = math.pi * i/n
    y[i] = math.sin(4.0 * x[i]) + math.sin(20.0 * x[i])
stddraw.setXscale(0, math.pi)
stddraw.setYscale(-2., 2.)
for i in range(n):
    stddraw.line(x[i], y[i], x[i+1], y[i+1])
    
stddraw.show()