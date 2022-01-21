import matplotlib.pyplot as plt
import numpy as np
  
x = np.linspace(0, 10*np.pi, 100)
y = np.sin(x)
  
plt.ion()
fig, ax = plt.subplots(3, sharex=True, sharey=True)
# fig = plt.figure()
# ax[0] = fig.add_subplot(111)
# ax[1] = fig.add_subplot(111)
# ax[2] = fig.add_subplot(111)
line1, = ax[0].plot(x, y, 'b-')
line2, = ax[1].plot(x, y, 'g-')
line3, = ax[2].plot(x, y, 'r-')
  
for phase in np.linspace(0, 10*np.pi, 1000):
    print(type(np.sin(0.5 * x + phase)))
    print(np.sin(0.5 * x + phase))
    line1.set_ydata(np.sin(0.5 * x + phase))
    line2.set_ydata(np.sin(0.5 * x + phase + 5))
    line3.set_ydata(np.sin(0.5 * x + phase/2))

    fig.canvas.draw()
    fig.canvas.flush_events()
