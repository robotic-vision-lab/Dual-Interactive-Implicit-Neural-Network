import torch.utils.benchmark as benchmark
import torch
from src.models.components.imsisr import IMSISR
from src.models.components.liif import LIIF
from src.models.components.metasr import MetaSR
from src.models.sr_module import BICUBIC_NET
from torch.nn.functional import avg_pool2d
bicubic_m = BICUBIC_NET().cuda()
metasr_m = MetaSR().cuda()
liif_m = LIIF().cuda()
imsisr_m = IMSISR(3, False).cuda()
num_threads = torch.get_num_threads()

@torch.no_grad()
def bicubic(x, size):
    return bicubic_m(x, (size, size))

@torch.no_grad()
def metasr(x, size):
    return metasr_m(x, (size, size))

@torch.no_grad()
def liif(x, size):
    return liif_m(x, (size, size))

@torch.no_grad()
def imsisr(x, size):
    return avg_pool2d(imsisr_m(x, (size, size)), 3, 1, 1)

x = torch.rand(1,3,48,48).cuda()

for size in [128, 256, 512]:
    tbicubic = benchmark.Timer(
        stmt='bicubic(x, size)',
        setup='from __main__ import bicubic',
        globals={'x': x, 'size': size},
        num_threads=num_threads)

    tmetasr = benchmark.Timer(
        stmt='metasr(x, size)',
        setup='from __main__ import metasr',
        globals={'x': x, 'size': size},
        num_threads=num_threads)
    
    tliif = benchmark.Timer(
        stmt='liif(x, size)',
        setup='from __main__ import liif',
        globals={'x': x, 'size': size},
        num_threads=num_threads)

    timsisr = benchmark.Timer(
        stmt='imsisr(x, size)',
        setup='from __main__ import imsisr',
        globals={'x': x, 'size': size},
        num_threads=num_threads)

    print(size, tbicubic.timeit(100))
    print(size, tmetasr.timeit(100))
    print(size, tliif.timeit(100))
    print(size, timsisr.timeit(100))