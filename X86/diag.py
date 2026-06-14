import numpy as np

def load(p, dt):
    with open(p, 'rb') as f:
        n = np.fromfile(f, dtype=np.int32, count=1)[0]
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        a = np.fromfile(f, dtype=dt, count=int(n)*int(d)).reshape(int(n), int(d))
    return a

base  = load('data/DEEP100K.base.100k.fbin', np.float32)
query = load('data/DEEP100K.query.fbin', np.float32)

# gt format: header(n,d) then [n*d int32 IDs][n*d float32 distances]
with open('data/DEEP100K.gt.query.100k.top100.bin', 'rb') as f:
    gn = int(np.fromfile(f, dtype=np.int32, count=1)[0])
    gd = int(np.fromfile(f, dtype=np.int32, count=1)[0])
    gt_id  = np.fromfile(f, dtype=np.int32,   count=gn*gd).reshape(gn, gd)
    gt_dst = np.fromfile(f, dtype=np.float32, count=gn*gd).reshape(gn, gd)
print('gt n=%d d=%d' % (gn, gd))
print('gt_id[0][:10]  =', gt_id[0][:10].tolist())
print('gt_dst[0][:5]  =', gt_dst[0][:5].tolist())
print('gt_id min/max  =', gt_id.min(), gt_id.max())
print('base row-norm mean/std =',
      np.linalg.norm(base[:2000], axis=1).mean(),
      np.linalg.norm(base[:2000], axis=1).std())

q  = query[0]
ip = base @ q
l2 = ((base - q) ** 2).sum(1)
ip_top = np.argsort(-ip)[:10]
l2_top = np.argsort(l2)[:10]
g = set(gt_id[0][:10].tolist())
print('IP-max top10 =', ip_top.tolist())
print('L2-min top10 =', l2_top.tolist())
print('IP overlap w/gt =', len(g & set(ip_top.tolist())),
      ' L2 overlap w/gt =', len(g & set(l2_top.tolist())))
