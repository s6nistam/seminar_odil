#!/usr/bin/env python3

import argparse
import pickle

import numpy as np

import odil
from odil import printlog
from odil.runtime import tf
import matplotlib.pyplot as plt


def plot_exact(
    domain,
    u_ref,
    path=None,
    umin=None,
    umax=None,
    transpose=False,
    dpi=300,
    aspect="auto",
    interpolation="nearest",
    cmap=None,
):
    if transpose:
        # Index zero drawn as vertical, rotate by 90 degrees counterclockwise.
        ix = 1
        iy = 0
        u_ref = u_ref.T
    else:
        # Index zero drawn as horizontal.
        ix = 0
        iy = 1
    extent = [domain.lower[ix], domain.upper[ix], domain.lower[iy], domain.upper[iy]]
    fig = plt.figure()
    # outer grid: left for main image, right for stacked slices
    outer = fig.add_gridspec(1, 1)
    xx, yy = domain.points_1d(ix, iy)
    xx, yy = np.array(xx), np.array(yy)
    xlim = (domain.lower[ix], domain.upper[ix])
    ylim = (domain.lower[iy], domain.upper[iy])
    if umin is None:
        umin = u_ref.min()
    if umax is None:
        umax = u_ref.max()
    if cmap is None:
        cmap = "viridis"
    ulim = (umin, umax)
    # main image occupies full height on the left
    main_gs = outer[0, 0].subgridspec(1, 1)
    ax = fig.add_subplot(main_gs[:, 0])
    ax.spines[:].set_visible(True)
    ax.spines[:].set_linewidth(0.25)
    ax.imshow(
        u_ref.T,
        interpolation=interpolation,
        cmap=cmap,
        vmin=ulim[0],
        vmax=ulim[1],
        extent=extent,
        origin="lower",
        aspect=aspect,
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$", labelpad=6)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # larger colorbar so it spans the main axes height
    fig.colorbar(ax.images[0], ax=ax, fraction=0.08, pad=0.02, ticks=[ulim[0], ulim[1]])

    if path is not None:
        fig.savefig(path, dpi=dpi, pad_inches=0.01)
        plt.close(fig)
    else:
        return fig

def get_exact(args, t, x, r, a):
    t = tf.Variable(t)
    x = tf.Variable(x)
    u = tf.zeros_like(x)
    with tf.GradientTape() as tape:
        u += (2 * r * np.pi * tf.exp(-(np.pi)**2 * r * t) * tf.sin(np.pi * x))/(a + tf.exp(-(np.pi)**2 * r * t) * tf.cos(np.pi * x))
    ut = tape.gradient(u, t).numpy()
    u = u.numpy()
    return u, ut

# def get_exact(args, t, x, r):
#     t = tf.Variable(t)
#     x = tf.Variable(x)
#     u = tf.zeros_like(x)
#     with tf.GradientTape() as tape:
#         u += (1/r)/(1 + t/r) * (x + tf.tan(0.5*x/(1 + t/r)))
#     ut = tape.gradient(u, t).numpy()
#     u = u.numpy()
#     return u, ut


def operator_burgers(ctx):
    extra = ctx.extra
    mod = ctx.mod
    args = extra.args
    dt, dx = ctx.step()
    it, ix = ctx.indices()
    nt, nx = ctx.size()
    ctx.points("x")

    def stencil_var(key):
        st = [
            ctx.field(key, 0, 0),
            ctx.field(key, -1, -1),
            ctx.field(key, -1, 1),
            ctx.field(key, -1, 0),
        ]
        return st

    left_utm = mod.roll(extra.left_u, 1, axis=0)
    right_utm = mod.roll(extra.right_u, 1, axis=0)

    def apply_bc_u(st):
        extrap = odil.core.extrap_quadh
        st[1] = mod.where(ix == 0, extrap(st[2], st[0], left_utm[:, None]), st[1])  #
        st[2] = mod.where(ix == nx - 1, extrap(st[1], st[0], right_utm[:, None]), st[2])  #
        return st

    u_st = stencil_var("u")
    apply_bc_u(u_st)
    u, uxm, uxp, utm = u_st

    u_t = (u - utm) / dt
    u_x = (uxp - uxm) / (2 * dx)
    u_xx = (uxm - 2 * utm + uxp) / (dx**2)

    fu = u_t + u * u_x - args.r * u_xx
    # fu = u_t + u * u_x - 1/args.r * u_xx

    u0 = extra.init_u + 0.5 * dt * extra.init_ut
    fu = mod.where(it == 0, (u - u0[None, :]) * args.kimp, fu)

    res = [("fu", fu)]

    return res


def get_uut(domain, init_u, uu):
    from odil.core import extrap_quad, extrap_quadh

    dt = domain.step("t")
    u = uu
    utm = np.roll(u, 1, axis=0)
    utp = np.roll(u, -1, axis=0)
    utm[0, :] = extrap_quadh(utp[0, :], u[0, :], init_u)
    utp[-1, :] = extrap_quad(u[-3, :], u[-2, :], u[-1, :])
    uut = (utp - utm) / (2 * dt)
    return uut


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--Nt", type=int, default=64, help="Grid size in t")
    parser.add_argument("--Nx", type=int, default=64, help="Grid size in x")
    parser.add_argument("--r", type=float, default=0.02, help="Reynolds number/viscosity parameter")
    parser.add_argument("--a", type=float, default=2, help="'a' parameter for the exact solution")
    parser.add_argument("--kimp", type=float, default=1, help="Factor to impose initial conditions")
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)
    parser.set_defaults(double=1)
    parser.set_defaults(multigrid=0)
    parser.set_defaults(outdir="out_burgers")
    parser.set_defaults(linsolver="direct")
    parser.set_defaults(optimizer="lbfgsb")
    # parser.set_defaults(optimizer="newton")
    # parser.set_defaults(optimizer="adam")
    parser.set_defaults(lr=0.001)
    parser.set_defaults(plotext="png", plot_title=1)
    parser.set_defaults(plot_every=1000, report_every=10, history_full=5, history_every=10, frames=100)
    # parser.set_defaults(plot_every=1, report_every=10, history_full=5, history_every=10, frames=5)
    return parser.parse_args()


def plot_func(problem, state, epoch, frame, cbinfo=None, plotext=None):
    from odil.plot import plot_1d

    domain = problem.domain
    extra = problem.extra
    mod = domain.mod
    args = extra.args

    if plotext is None:
        plotext = args.plotext

    title0 = "u epoch={:05d}".format(epoch) if args.plot_title else None
    title1 = "ut epoch={:05d}".format(epoch) if args.plot_title else None
    path0 = "u_{:05d}.{}".format(frame, plotext)
    path1 = "ut_{:05d}.{}".format(frame, plotext)
    printlog(path0, path1)

    ref_u, ref_ut = extra.ref_u, extra.ref_ut

    state_u = np.array(domain.field(state, "u"))
    state_ut = get_uut(domain, extra.init_u, state_u)

    if args.dump_data:
        path = "data_{:05d}.pickle".format(frame)
        d = dict()
        d["upper"] = domain.upper
        d["lower"] = domain.lower
        d["cshape"] = domain.cshape
        d["state_u"] = state_u
        d["state_ut"] = state_ut
        d["ref_u"] = ref_u
        d["ref_ut"] = ref_ut
        d = odil.core.struct_to_numpy(mod, d)
        with open(path, "wb") as f:
            pickle.dump(d, f)

    umax = max(abs(np.max(ref_u)), abs(np.min(ref_u)))
    plot_1d(
        domain,
        extra.ref_u,
        state_u,
        path=path0,
        title=title0,
        cmap="RdBu_r",
        nslices=5,
        transpose=True,
        umin=-umax,
        umax=umax,
    )

    plot_exact(
        domain,
        u_ref=ref_u,
        path="exact_burgers.svg",
        cmap="RdBu_r",
        transpose=True,
        umin=-umax,
        umax=umax,
    )

    umax = max(abs(np.max(ref_ut)), abs(np.min(ref_ut)))
    plot_1d(
        domain,
        ref_ut,
        state_ut,
        path=path1,
        title=title1,
        cmap="RdBu_r",
        nslices=5,
        transpose=True,
        umin=-umax,
        umax=umax,
    )


def get_error(domain, extra, state, key):
    if key == "u":
        state_u = domain.field(state, key)
        ref_u = extra.ref_u
        return np.sqrt(np.mean((state_u - ref_u) ** 2))
    return None


def history_func(problem, state, epoch, history, cbinfo):
    domain = problem.domain
    extra = problem.extra
    for key in ["u", "k"]:
        error = get_error(domain, extra, state, key)
        if error is not None:
            history.append("error_" + key, error)


def report_func(problem, state, epoch, cbinfo):
    domain = problem.domain
    extra = problem.extra
    res = dict()
    for key in ["u", "k"]:
        error = get_error(domain, extra, state, key)
        if error is not None:
            res[key] = error
    printlog("error: " + ", ".join("{}:{:.5g}".format(*item) for item in res.items()))


def make_problem(args):
    dtype = np.float64 if args.double else np.float32
    domain = odil.Domain(
        cshape=(args.Nt, args.Nx),
        dimnames=("t", "x"),
        # lower=(0, -3),
        # upper=(1, 3),
        lower=(0, 0),
        upper=(10, 1),
        multigrid=args.multigrid,
        dtype=dtype,
    )
    if domain.multigrid:
        printlog("multigrid levels:", domain.mg_cshapes)
    mod = domain.mod
    # Evaluate exact solution, boundary and initial conditions.
    tt, xx = domain.points()
    t1, x1 = domain.points_1d()
    ref_u, ref_ut = get_exact(args, tt, xx, args.r, args.a)
    left_u, _ = get_exact(args, t1, t1 * 0 + domain.lower[1], args.r, args.a)
    right_u, _ = get_exact(args, t1, t1 * 0 + domain.upper[1], args.r, args.a)
    init_u, init_ut = get_exact(args, x1 * 0 + domain.lower[0], x1, args.r, args.a)

    extra = argparse.Namespace()

    def add_extra(d, *keys):
        for key in keys:
            setattr(extra, key, d[key])

    add_extra(locals(), "args", "ref_u", "ref_ut", "left_u", "right_u", "init_u", "init_ut")

    state = odil.State()
    state.fields["u"] = np.zeros(domain.cshape)
    # state.fields["u"] = np.random.normal(size=domain.cshape)
    state = domain.init_state(state)
    problem = odil.Problem(operator_burgers, domain, extra)
    return problem, state


def main():
    args = parse_args()
    odil.setup_outdir(args)
    problem, state = make_problem(args)
    callback = odil.make_callback(
        problem, args, plot_func=plot_func, history_func=history_func, report_func=report_func
    )
    try:
        # arrays, optinfo = odil.util.optimize(args, args.optimizer, problem, state, callback)
        arrays, optinfo = odil.util.optimize(args, args.optimizer, problem, state, callback, factr=10000)
    except odil.optimizer.EarlyStopError as e:
        print(f"Early stop: {e}")
        plot_func(problem, state, args.epochs, args.frames)


    with open("done", "w"):
        pass


if __name__ == "__main__":
    main()
