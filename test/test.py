import numpy as np

state=np.zeros((3,3),np.int8)
state[1][1]=33
where=(1,1)
print(state[where])