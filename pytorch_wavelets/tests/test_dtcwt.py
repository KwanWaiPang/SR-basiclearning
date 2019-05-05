import pytest

import numpy as np
from Transform2d_np import Transform2d as Transform2d_np
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from pytorch_wavelets.dtcwt.coeffs import biort as _biort, qshift as _qshift
import datasets
import torch
import py3nvml
from contextlib import contextmanager
PRECISION_FLOAT = 3
PRECISION_DOUBLE = 7

HAVE_GPU = torch.cuda.is_available()
if HAVE_GPU:
    dev = torch.device('cuda')
else:
    dev = torch.device('cpu')


@contextmanager
def set_double_precision():
    old_prec = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        yield
    finally:
        torch.set_default_dtype(old_prec)


def setup():
    global barbara, barbara_t
    global bshape, bshape_half
    global ch
    py3nvml.grab_gpus(1, gpu_fraction=0.5)
    barbara = datasets.barbara()
    barbara = (barbara/barbara.max()).astype('float32')
    barbara = barbara.transpose([2, 0, 1])
    bshape = list(barbara.shape)
    bshape_half = bshape[:]
    bshape_half[1] //= 2
    barbara_t = torch.unsqueeze(
        torch.tensor(barbara, dtype=torch.float32, device=dev), dim=0)
    ch = barbara_t.shape[1]


def test_barbara_loaded():
    assert barbara.shape == (3, 512, 512)
    assert barbara.min() >= 0
    assert barbara.max() <= 1
    assert barbara.dtype == np.float32
    assert list(barbara_t.shape) == [1, 3, 512, 512]


def test_simple():
    xfm = DTCWTForward(J=3).to(dev)
    Yl, Yh = xfm(barbara_t)
    assert len(Yl.shape) == 4
    assert len(Yh) == 3
    assert Yh[0].shape[-1] == 2


def test_specific_wavelet():
    xfm = DTCWTForward(J=3, biort='antonini', qshift='qshift_06').to(dev)
    Yl, Yh = xfm(barbara_t)
    assert len(Yl.shape) == 4
    assert len(Yh) == 3
    assert Yh[0].shape[-1] == 2


def test_odd_rows():
    xfm = DTCWTForward(J=3).to(dev)
    Yl, Yh = xfm(barbara_t[:,:,:509])


def test_odd_cols():
    xfm = DTCWTForward(J=3).to(dev)
    Yl, Yh = xfm(barbara_t[:,:,:,:509])


def test_odd_rows_and_cols():
    xfm = DTCWTForward(J=3).to(dev)
    Yl, Yh = xfm(barbara_t[:,:,:509,:509])


@pytest.mark.parametrize("J, o_before_c", [
    (1,False),(1,True),(2, False), (2,True),
    (3, False),(3, True),(4, False), (4, True),
    (5, False), (5, True)
])
def test_fwd(J, o_before_c):
    X = 100*np.random.randn(3, 5, 100, 100)
    Xt = torch.tensor(X, dtype=torch.get_default_dtype(), device=dev)
    xfm = DTCWTForward(J=J, o_before_c=o_before_c).to(dev)
    Yl, Yh = xfm(Xt)
    f1 = Transform2d_np()
    yl, yh = f1.forward(X, nlevels=J)

    np.testing.assert_array_almost_equal(
        Yl.cpu(), yl, decimal=PRECISION_FLOAT)
    for i in range(len(yh)):
        if o_before_c:
            np.testing.assert_array_almost_equal(
                Yh[i][...,0].cpu().transpose(2,1), yh[i].real,
                decimal=PRECISION_FLOAT)
            np.testing.assert_array_almost_equal(
                Yh[i][...,1].cpu().transpose(2,1), yh[i].imag,
                decimal=PRECISION_FLOAT)
        else:
            np.testing.assert_array_almost_equal(
                Yh[i][...,0].cpu(), yh[i].real, decimal=PRECISION_FLOAT)
            np.testing.assert_array_almost_equal(
                Yh[i][...,1].cpu(), yh[i].imag, decimal=PRECISION_FLOAT)


@pytest.mark.parametrize("J, o_before_c", [
    (1,False),(1,True),(2, False), (2,True),
    (3, False),(3, True),(4, False), (4, True),
    (5, False), (5, True)
])
def test_fwd_double(J, o_before_c):
    with set_double_precision():
        X = 100*np.random.randn(3, 5, 100, 100)
        Xt = torch.tensor(X, dtype=torch.get_default_dtype(), device=dev)
        xfm = DTCWTForward(J=J, o_before_c=o_before_c).to(dev)
        Yl, Yh = xfm(Xt)
    assert Yl.dtype == torch.float64
    f1 = Transform2d_np()
    yl, yh = f1.forward(X, nlevels=J)

    np.testing.assert_array_almost_equal(
        Yl.cpu(), yl, decimal=PRECISION_DOUBLE)
    for i in range(len(yh)):
        if o_before_c:
            np.testing.assert_array_almost_equal(
                Yh[i][...,0].cpu().transpose(2,1), yh[i].real,
                decimal=PRECISION_DOUBLE)
            np.testing.assert_array_almost_equal(
                Yh[i][...,1].cpu().transpose(2,1), yh[i].imag,
                decimal=PRECISION_DOUBLE)
        else:
            np.testing.assert_array_almost_equal(
                Yh[i][...,0].cpu(), yh[i].real, decimal=PRECISION_DOUBLE)
            np.testing.assert_array_almost_equal(
                Yh[i][...,1].cpu(), yh[i].imag, decimal=PRECISION_DOUBLE)


@pytest.mark.parametrize("J, o_before_c", [
    (1,False),(1,True),(2, False), (2,True),
    (3, False),(3, True),(4, False), (4, True),
    (5, False), (5, True)
])
def test_fwd_skip_hps(J, o_before_c):
    X = 100*np.random.randn(3, 5, 100, 100)
    # Randomly turn on/off the highpass outputs
    hps = np.random.binomial(size=J, n=1,p=0.5).astype('bool')
    xfm = DTCWTForward(J=J, skip_hps=hps, o_before_c=o_before_c).to(dev)
    Yl, Yh = xfm(torch.tensor(X, dtype=torch.float32, device=dev))
    f1 = Transform2d_np()
    yl, yh = f1.forward(X, nlevels=J)

    np.testing.assert_array_almost_equal(
        Yl.cpu(), yl, decimal=PRECISION_FLOAT)
    for j in range(J):
        if hps[j]:
            assert Yh[j].shape == torch.Size([0])
        else:
            if o_before_c:
                np.testing.assert_array_almost_equal(
                    Yh[j][...,0].cpu().transpose(2,1), yh[j].real,
                    decimal=PRECISION_FLOAT)
                np.testing.assert_array_almost_equal(
                    Yh[j][...,1].cpu().transpose(2,1), yh[j].imag,
                    decimal=PRECISION_FLOAT)
            else:
                np.testing.assert_array_almost_equal(
                    Yh[j][...,0].cpu(), yh[j].real, decimal=PRECISION_FLOAT)
                np.testing.assert_array_almost_equal(
                    Yh[j][...,1].cpu(), yh[j].imag, decimal=PRECISION_FLOAT)


@pytest.mark.parametrize("scales", [
    (True,), (True, True), (True, True, True), (True, True, True, True),
    (True, False, True), (False, True, True), (True, True, False),
    (True, True, False, True)])
def test_fwd_include_scale(scales):
    X = 100*np.random.randn(3, 5, 100, 100)
    # Randomly turn on/off the highpass outputs
    J = len(scales)
    xfm = DTCWTForward(J=J, include_scale=scales).to(dev)
    Ys, Yh = xfm(torch.tensor(X, dtype=torch.float32, device=dev))
    f1 = Transform2d_np()
    yl, yh, ys = f1.forward(X, nlevels=J, include_scale=True)

    for j in range(J):
        if not scales[j]:
            assert Ys[j].shape == torch.Size([0])
        else:
            np.testing.assert_array_almost_equal(
                Ys[j].cpu(), ys[j], decimal=PRECISION_FLOAT)


@pytest.mark.parametrize("scales", [
    (True,), (True, True), (True, True, True), (True, True, True, True),
    (True, False, True), (False, True, True), (True, True, False),
    (True, True, False, True)])
def test_bwd_include_scale(scales):
    X = 100*np.random.randn(3, 5, 100, 100)
    # Randomly turn on/off the highpass outputs
    J = len(scales)
    xfm = DTCWTForward(J=J, include_scale=scales).to(dev)
    Ys, Yh = xfm(torch.tensor(X, dtype=torch.float32, requires_grad=True,
                              device=dev))
    f1 = Transform2d_np()
    yl, yh, ys = f1.forward(X, nlevels=J, include_scale=True)

    for ys in Ys:
        ys.backward(torch.ones_like(ys),retain_graph=True)


@pytest.mark.parametrize("J, o_before_c", [
    (1,False),(1,True),(2, False), (2,True),
    (3, False),(3, True),(4, False), (4, True),
    (5, False), (5, True)
])
def test_inv(J, o_before_c):
    Yl = 100*np.random.randn(3, 5, 64, 64)
    Yhr = [np.random.randn(3, 5, 6, 2**j, 2**j) for j in range(4+J,4,-1)]
    Yhi = [np.random.randn(3, 5, 6, 2**j, 2**j) for j in range(4+J,4,-1)]
    Yh1 = [yhr + 1j*yhi for yhr, yhi in zip(Yhr, Yhi)]
    if o_before_c:
        Yh2 = [torch.tensor(np.stack((yhr, yhi), axis=-1),
                            dtype=torch.float32, device=dev).transpose(1,2)
               for yhr, yhi in zip(Yhr, Yhi)]
    else:
        Yh2 = [torch.tensor(np.stack((yhr, yhi), axis=-1),
                            dtype=torch.float32, device=dev)
               for yhr, yhi in zip(Yhr, Yhi)]

    ifm = DTCWTInverse(J=J, o_before_c=o_before_c).to(dev)
    X = ifm((torch.tensor(Yl, dtype=torch.float32, device=dev), Yh2))
    f1 = Transform2d_np()
    x = f1.inverse(Yl, Yh1)

    np.testing.assert_array_almost_equal(
        X.cpu(), x, decimal=PRECISION_FLOAT)


@pytest.mark.parametrize("J, o_before_c", [
    (1,False),(1,True),(2, False), (2,True),
    (3, False),(3, True),(4, False), (4, True),
    (5, False), (5, True)
])
def test_inv_skip_hps(J, o_before_c):
    hps = np.random.binomial(size=J, n=1,p=0.5).astype('bool')
    Yl = 100*np.random.randn(3, 5, 64, 64)
    Yhr = [np.random.randn(3, 5, 6, 2**j, 2**j) for j in range(4+J,4,-1)]
    Yhi = [np.random.randn(3, 5, 6, 2**j, 2**j) for j in range(4+J,4,-1)]
    Yh1 = [yhr + 1j*yhi for yhr, yhi in zip(Yhr, Yhi)]
    if o_before_c:
        Yh2 = [torch.tensor(np.stack((yhr, yhi), axis=-1),
                            dtype=torch.float32, device=dev).transpose(1,2)
               for yhr, yhi in zip(Yhr, Yhi)]
    else:
        Yh2 = [torch.tensor(np.stack((yhr, yhi), axis=-1),
                            dtype=torch.float32, device=dev)
               for yhr, yhi in zip(Yhr, Yhi)]
    for j in range(J):
        if hps[j]:
            Yh2[j] = torch.tensor([])
            Yh1[j] = np.zeros_like(Yh1[j])

    ifm = DTCWTInverse(J=J, o_before_c=o_before_c).to(dev)
    X = ifm((torch.tensor(Yl, dtype=torch.float32, requires_grad=True,
                          device=dev), Yh2))
    # Also test giving None instead of an empty tensor
    for j in range(J):
        if hps[j]:
            Yh2[j] = None
    X2 = ifm((torch.tensor(Yl, dtype=torch.float32, device=dev), Yh2))
    f1 = Transform2d_np()
    x = f1.inverse(Yl, Yh1)

    np.testing.assert_array_almost_equal(
        X.detach().cpu(), x, decimal=PRECISION_FLOAT)
    np.testing.assert_array_almost_equal(
        X2.cpu(), x, decimal=PRECISION_FLOAT)

    # Test gradients are ok
    X.backward(torch.ones_like(X))


# Test end to end with numpy inputs
@pytest.mark.parametrize("biort,qshift,size,J", [
    ('antonini','qshift_a', (128,128), 3),
    ('antonini','qshift_a', (126,126), 3),
    ('legall','qshift_a', (99,100), 4),
    ('near_sym_a','qshift_c', (104, 101), 2),
    ('near_sym_b','qshift_d', (126, 126), 3),
])
def test_end2end(biort, qshift, size, J):
    im = np.random.randn(5,6,*size).astype('float32')
    imt = torch.tensor(im, dtype=torch.float32, requires_grad=True, device=dev)
    xfm = DTCWTForward(J=J, biort=biort, qshift=qshift).to(dev)
    Yl, Yh = xfm(imt)
    ifm = DTCWTInverse(J=J, biort=biort, qshift=qshift).to(dev)
    y = ifm((Yl, Yh))

    # Compare with numpy results
    f_np = Transform2d_np(biort=biort, qshift=qshift)
    yl, yh = f_np.forward(im, nlevels=J)
    y2 = f_np.inverse(yl, yh)

    np.testing.assert_array_almost_equal(y.detach().cpu(), y2, decimal=PRECISION_FLOAT)

    # Test gradients are ok
    y.backward(torch.ones_like(y))


# Test gradients
@pytest.mark.parametrize("biort,qshift,size,J", [
    ('antonini','qshift_a', (128,128), 3),
    ('antonini','qshift_a', (64,64), 3),
    ('legall','qshift_a', (240,240), 4),
    ('near_sym_a','qshift_c', (100, 100), 2),
    ('near_sym_b','qshift_d', (120, 120), 3),
])
def test_gradients_fwd(biort, qshift, size, J):
    """ Gradient of forward function should be inverse function with filters
    swapped """
    im = np.random.randn(5,6,*size).astype('float32')
    imt = torch.tensor(im, dtype=torch.float32, requires_grad=True, device=dev)
    xfm = DTCWTForward(biort=biort, qshift=qshift, J=J).to(dev)
    h0o, g0o, h1o, g1o = _biort(biort)
    h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(qshift)
    xfm_grad = DTCWTInverse(J=J, biort=(h0o[::-1], h1o[::-1]),
                            qshift=(h0a[::-1], h0b[::-1], h1a[::-1], h1b[::-1])
                            ).to(dev)
    Yl, Yh = xfm(imt)
    Ylg = torch.randn(*Yl.shape, device=dev)
    Yl.backward(Ylg, retain_graph=True)
    ref = xfm_grad((Ylg, [None,]*J))
    np.testing.assert_array_almost_equal(imt.grad.detach().cpu(), ref.cpu())
    for j, y in enumerate(Yh):
        imt.grad.zero_()
        g = torch.randn(*y.shape, device=dev)
        y.backward(g, retain_graph=True)
        hps = [None,] * J
        hps[j] = g
        ref = xfm_grad((torch.zeros_like(Yl), hps))
        np.testing.assert_array_almost_equal(imt.grad.detach().cpu(), ref.cpu())


@pytest.mark.parametrize("biort,qshift,size,J", [
    ('antonini','qshift_a', (128,128), 3),
    ('antonini','qshift_a', (64,64), 3),
    ('legall','qshift_a', (240,240), 4),
    ('near_sym_a','qshift_c', (100, 100), 2),
    ('near_sym_b','qshift_d', (120, 120), 3),
])
def test_gradients_inv(biort, qshift, size, J):
    """ Gradient of forward function should be inverse function with filters
    swapped """
    im = np.random.randn(5,6,*size).astype('float32')
    imt = torch.tensor(im, dtype=torch.float32, device=dev)
    ifm = DTCWTInverse(biort=biort, qshift=qshift, J=J).to(dev)
    h0o, g0o, h1o, g1o = _biort(biort)
    h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(qshift)
    ifm_grad = DTCWTForward(J=J, biort=(g0o[::-1], g1o[::-1]),
                            qshift=(g0a[::-1], g0b[::-1], g1a[::-1], g1b[::-1])
                            ).to(dev)
    yl, yh = ifm_grad(imt)
    g = torch.randn(*imt.shape, device=dev)
    ylv = torch.randn(*yl.shape, requires_grad=True, device=dev)
    yhv = [torch.randn(*h.shape, requires_grad=True, device=dev) for h in yh]
    Y = ifm((ylv, yhv))
    Y.backward(g)

    # Check the lowpass gradient is the same
    ref_lp, ref_bp = ifm_grad(g)
    np.testing.assert_array_almost_equal(ylv.grad.detach().cpu(), ref_lp.cpu())
    # check the bandpasses are the same
    for y, ref in zip(yhv, ref_bp):
        np.testing.assert_array_almost_equal(y.grad.detach().cpu(), ref.cpu())
