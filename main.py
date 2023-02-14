from reader import Reader
from denclue2D import Denclue2D, H, K, DELTA, XI

if __name__ == "__main__":
    for i in range(1, 16):
        r = Reader("D:\Code\DENCLUE\data/%d.in" % i)
        x, y = r.read()
        dc = Denclue2D(x, y)
        dc.work()
        dc.render_dens_fig("D:\Code\DENCLUE\output/dens_fig/%d.png" % i)
