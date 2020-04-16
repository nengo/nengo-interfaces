import numpy as np
import matplotlib.pyplot as plt

brent = np.load('brent.npz')
brent2 = np.load('brent2.npz')
# noise removed from self.pos
brent3 = np.load('brent3.npz')
mine = np.load('mine.npz')
datas = [brent, brent3, mine]
ls = ['-', '--', '-.']
cols = ['r', '--g', 'b', '--k']

fig1 = plt.figure()
a1 = []
a1.append(fig1.add_subplot(711))
a1.append(fig1.add_subplot(712))
a1.append(fig1.add_subplot(713))
a1.append(fig1.add_subplot(714))

a2 = []
a2.append(fig1.add_subplot(715))
a2.append(fig1.add_subplot(716))
a2.append(fig1.add_subplot(717))


for jj, data in enumerate(datas):
    u_track = data['u']
    pos_track = data['pos']
    for ii, u in enumerate(u_track.T):
        a1[ii].set_title('u%i'%ii)
        a1[ii].plot(u,
                # linestyle=ls[jj],
                cols[jj])
    for ii, pos in enumerate(pos_track.T):
        a2[ii].set_title('pos%i'%ii)
        a2[ii].plot(pos,
                # linestyle=ls[jj],
                cols[jj])
plt.show()
