"""
Source code: https://stackoverflow.com/questions/18390461/scroll-backwards-and-forwards-through-matplotlib-plots
"""


import numpy as np
import matplotlib.pyplot as plt

# define your x and y arrays to be plotted
number_of_points = 25
t = np.linspace(start=0, stop=2*np.pi, num=number_of_points)
number_of_intervals = 50
delta_t = np.linspace(start = 0, stop = 2*np.pi, num = number_of_intervals)
plots = []
for i in range(number_of_intervals) :
    temp_y = np.cos(t+delta_t[i-1])
    temp_array = (t,temp_y)
    plots.append(temp_array)


# y1 = np.cos(t+0.1)
# y2 = np.cos(t+0.2)
# y3 = np.cos(t+0.3)
# plots = [(t,y1), (t,y2), (t,y3)]

# now the real code :) 
curr_pos = 0

def key_event(e):
    global curr_pos

    if e.key == "right":
        curr_pos = curr_pos + 1
    elif e.key == "left":
        curr_pos = curr_pos - 1
    else:
        return
    curr_pos = curr_pos % len(plots)

    ax.cla()
    ax.plot(plots[curr_pos][0], plots[curr_pos][1])
    fig.canvas.draw()

fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', key_event)
ax = fig.add_subplot(111)
ax.plot(t,temp_y)
plt.show()