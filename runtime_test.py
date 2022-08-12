import timeit
import torch
from src.models.components.imsisr import IMSISR
from src.models.components.liif import LIIF
from src.models.components.metasr import MetaSR
from src.models.sr_module import BICUBIC_NET

bicubic_m = BICUBIC_NET()
metasr_m = MetaSR()
liif_m = LIIF()
imsisr_m = IMSISR(3, False)

def bicubic(x, size):
    return bicubic_m(x, (size, size))

def metasr(x, size):
    return metasr_m(x, (size, size))

def liif(x, size):
    return liif_m(x, (size, size))

def imsisr(x, size):
    return imsisr_m(x, (size, size))

x = torch.rand(16,3,48,48)

for size in [128, 256, 512]:
    tbicubic = timeit.Timer(
        stmt='bicubic(x, size)',
        setup='from __main__ import bicubic',
        globals={'x': x, 'size': size})

    tmetasr = timeit.Timer(
        stmt='metasr(x, size)',
        setup='from __main__ import metasr',
        globals={'x': x, 'size': size})
    
    tliif = timeit.Timer(
        stmt='liif(x, size)',
        setup='from __main__ import liif',
        globals={'x': x, 'size': size})

    timsisr = timeit.Timer(
        stmt='imsisr(x, size)',
        setup='from __main__ import imsisr',
        globals={'x': x, 'size': size})

print(f'bicubic: {tbicubic.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'metasr: {tmetasr.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'liif: {tliif.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'imsisr: {timsisr.timeit(100) / 100 * 1e6:>5.1f} us')