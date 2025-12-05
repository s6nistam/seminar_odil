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

def get_exact(args, t, x):
    t = tf.Variable(t)
    x = tf.Variable(x)
    u = tf.zeros_like(x)
    with tf.GradientTape() as tape:
        ii = [1, 2, 3, 4, 5]
        for i in ii:
            k = i * np.pi
            u += tf.cos((x - t + 0.5) * k)
            u += tf.cos((x + t - 0.5) * k)
        u /= 2 * len(ii)
    ut = tape.gradient(u, t).numpy()
    u = u.numpy()
    return u, ut

def operator_wave(ctx):
    extra = ctx.extra
    mod = ctx.mod
    args = extra.args
    dt, dx = ctx.step()
    it, ix = ctx.indices()
    nt, nx = ctx.size()
    ctx.points("x")

    def stencil_var(key):
        st = [
            ctx.field(key),
            ctx.field(key, -1, 0),
            ctx.field(key, -2, 0),
            ctx.field(key, -1, -1),
            ctx.field(key, -1, 1),
        ]
        return st

    left_utm = mod.roll(extra.left_u, 1, axis=0)
    right_utm = mod.roll(extra.right_u, 1, axis=0)

    def apply_bc_u(st):
        extrap = odil.core.extrap_quadh
        st[3] = mod.where(ix == 0, extrap(st[4], st[1], left_utm[:, None]), st[3])  #
        st[4] = mod.where(ix == nx - 1, extrap(st[3], st[1], right_utm[:, None]), st[4])  #
        return st

    u_st = stencil_var("u")
    apply_bc_u(u_st)
    u, utm, utmm, uxm, uxp = u_st

    u_t_tm = (u - utm) / dt
    u_t_tmm = (utm - utmm) / dt
    u_t_tmm = mod.where(it == 1, extra.init_ut[None, :], u_t_tmm)

    u_tt = (u_t_tm - u_t_tmm) / dt
    u_xx = (uxm - 2 * utm + uxp) / (dx**2)

    fu = u_tt - u_xx

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
    parser.add_argument("--kimp", type=float, default=1, help="Factor to impose initial conditions")
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)
    parser.set_defaults(double=1)
    parser.set_defaults(multigrid=0)
    parser.set_defaults(outdir="out_wave")
    parser.set_defaults(linsolver="direct")
    # parser.set_defaults(optimizer="lbfgsb")
    parser.set_defaults(optimizer="newton")
    # parser.set_defaults(optimizer="adam")
    parser.set_defaults(lr=0.001)
    parser.set_defaults(plotext="png", plot_title=1)
    # parser.set_defaults(plotext="svg", plot_title=1)
    parser.set_defaults(plot_every=1, report_every=10, history_full=5, history_every=10, frames=10)
    return parser.parse_args()


def plot_func(problem, state, epoch, frame, cbinfo=None):
    from odil.plot import plot_1d_fd

    domain = problem.domain
    extra = problem.extra
    mod = domain.mod
    args = extra.args

    title0 = "u epoch={:05d}".format(epoch) if args.plot_title else None
    title1 = "ut epoch={:05d}".format(epoch) if args.plot_title else None
    path0 = "u_{:05d}.{}".format(frame, args.plotext)
    path1 = "ut_{:05d}.{}".format(frame, args.plotext)
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


    global u_fd
    umax = max(abs(np.max(ref_u)), abs(np.min(ref_u)))
    plot_1d_fd(
        domain,
        u_ref=ref_u,
        u_fd=u_fd,
        u_state=state_u,
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
        path="exact_wave.svg",
        cmap="RdBu_r",
        transpose=True,
        umin=-umax,
        umax=umax,
    )


    global ut_fd
    umax = max(abs(np.max(ref_ut)), abs(np.min(ref_ut)))
    plot_1d_fd(
        domain,
        ref_ut,
        ut_fd,
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

def get_error_t_end(domain, extra, state, key):
    if key == "u":
        state_u = domain.field(state, key)
        ref_u = extra.ref_u
        return np.sqrt(np.mean((state_u[-1,:] - ref_u[-1,:]) ** 2))
    return None

def get_error_fd(domain):
    Nt, Nx = domain.cshape
    t_lower, x_lower = domain.lower
    t_upper, x_upper = domain.upper
    dt, dx = domain.step()
    # x = np.linspace(x_lower, x_upper, Nx)
    # t = np.linspace(t_lower, t_upper, Nt)
    t, x = domain.points_1d()
    global u_fd
    U_exact_final, _ = get_exact([], x * 0 + t_upper - 0.5 * dt, x)
    
    # Calculate errors
    u_error = np.abs(u_fd[-1,:] - U_exact_final)
    max_u_error = np.max(u_error)
    return max_u_error


def history_func(problem, state, epoch, history, cbinfo):
    domain = problem.domain
    extra = problem.extra
    for key in ["u", "k"]:
        # error = get_error_t_end(domain, extra, state, key)
        error = get_error(domain, extra, state, key)
        if error is not None:
            history.append("error_" + key, error)
    history.append("error_" + "u_fd", get_error_fd(domain))


def report_func(problem, state, epoch, cbinfo):
    domain = problem.domain
    extra = problem.extra
    res = dict()
    for key in ["u", "k"]:
        error = get_error(domain, extra, state, key)
        if error is not None:
            res[key] = error
    res["u_fd"] = get_error_fd(domain)
    printlog("error: " + ", ".join("{}:{:.5g}".format(*item) for item in res.items()))


def make_problem(args):
    dtype = np.float64 if args.double else np.float32
    domain = odil.Domain(
        cshape=(args.Nt, args.Nx),
        dimnames=("t", "x"),
        lower=(0, -1),
        upper=(1, 1),
        multigrid=args.multigrid,
        dtype=dtype,
    )
    if domain.multigrid:
        printlog("multigrid levels:", domain.mg_cshapes)
    mod = domain.mod
    # Evaluate exact solution, boundary and initial conditions.
    tt, xx = domain.points()
    t1, x1 = domain.points_1d()
    ref_u, ref_ut = get_exact(args, tt, xx)
    # print(ref_u.shape, ref_ut.shape, "ref")
    left_u, _ = get_exact(args, t1, t1 * 0 + domain.lower[1])
    right_u, _ = get_exact(args, t1, t1 * 0 + domain.upper[1])
    init_u, init_ut = get_exact(args, x1 * 0 + domain.lower[0], x1)

    extra = argparse.Namespace()

    def add_extra(d, *keys):
        for key in keys:
            setattr(extra, key, d[key])

    add_extra(locals(), "args", "ref_u", "ref_ut", "left_u", "right_u", "init_u", "init_ut")

    state = odil.State()
    # state.fields["u"], _ = solve_fd_extrap(domain)
    # state.fields["u"], _ = solve_fd_dirichlet(domain)
    state.fields["u"] = np.zeros(domain.cshape)
    # state.fields["u"] = np.array([[1 if i == 0 else 0 for j in range(domain.cshape[1])]for i in range(domain.cshape[0])])
    # state.fields["u"] = np.random.normal(size=domain.cshape)
    state = domain.init_state(state)
    problem = odil.Problem(operator_wave, domain, extra)
    return problem, state

def solve_fd_strong_dirichlet(domain):
    # Unpack domain parameters
    Nt, Nx = domain.cshape
    t_lower, x_lower = domain.lower
    t_upper, x_upper = domain.upper
    dt, dx = domain.step()
    
    # Create grids
    t, x = domain.points_1d()
    
    u0, ut0 = get_exact([], x * 0 + t_lower, x)
    left_u, _ = get_exact([], t, t * 0 + x_lower + 0.5 * dx)
    right_u, _ = get_exact([], t, t * 0 + x_upper - 0.5 * dx)
    
    # Initialize solution arrays
    u = np.zeros((Nt, Nx))
    ut = np.zeros((Nt, Nx))
    
    u[0, :], ut[0, :] = get_exact([], x * 0 + t_lower + 0.5 * dt, x)

    # Apply strong Dirichlet boundary conditions
    u[:, 0] = left_u
    u[:, -1] = right_u
    
    # Compute first time step using exact initial derivative    
    # Interior points for n=1 using Taylor expansion
    for i in range(1, Nx - 1):
        # u(x, Δt) ≈ u(x,0) + Δt*u_t(x,0) + (Δt²/2)*u_xx(x,0)
        u[1, i] = u[0, i] + dt * ut[0, i] + (dt**2 / (2 * dx**2)) * (u[0, i+1] - 2*u[0, i] + u[0, i-1])
    
    for n in range(1, Nt - 1):
        # Update interior points using finite differences
        for i in range(1, Nx - 1):
            u[n+1, i] = 2*u[n, i] - u[n-1, i] + (dt**2 / dx**2) * (u[n, i+1] - 2*u[n, i] + u[n, i-1])

    return u, get_uut(domain, u0, u)

def solve_fd_ghosts_cells(domain):
    # Unpack domain parameters
    Nt, Nx = domain.cshape
    t_lower, x_lower = domain.lower
    t_upper, x_upper = domain.upper
    dt, dx = domain.step()
    
    # Create grids
    t, x = domain.points_1d()
    # print(t[-1], x[-1], "points")

    
    u0, ut0 = get_exact([], x * 0 + t_lower, x)
    left_u, _ = get_exact([], t, t * 0 + x_lower)
    right_u, _ = get_exact([], t, t * 0 + x_upper)
    
    # Initialize solution arrays
    u = np.zeros((Nt, Nx))
    ut = np.zeros((Nt, Nx))
    u[0, :], ut[0, :] = get_exact([], x * 0 + t_lower + 0.5 * dt, x)
    
    # Apply Dirichlet boundary conditions using ghost cells
    extrap = odil.core.extrap_quadh
    ghost_left = np.zeros(Nt)
    ghost_right = np.zeros(Nt)
    ghost_left[0] = extrap(u[0, 1], u[0, 0], left_u[0])
    ghost_right[0] = extrap(u[0, -2], u[0, -1], right_u[0])

    # Compute first time step using exact initial derivative    
    # Interior points for n=1 using Taylor expansion

    u[1, 0] = u[0, 0] + dt * ut[0, 0] + (dt**2 / (2 * dx**2)) * (u[0, 1] - 2*u[0, 0] + ghost_left[0])
    for i in range(1, Nx - 1):
        # u(x, Δt) ≈ u(x,0) + Δt*u_t(x,0) + (Δt²/2)*u_xx(x,0)
        u[1, i] = u[0, i] + dt * ut[0, i] + (dt**2 / (2 * dx**2)) * (u[0, i+1] - 2*u[0, i] + u[0, i-1])
    u[1, -1] = u[0, -1] + dt * ut[0, -1] + (dt**2 / (2 * dx**2)) * (ghost_right[0] - 2*u[0, -1] + u[0, -2])
    
    for n in range(1, Nt - 1):
        ghost_left[n] = extrap(u[n, 1], u[n, 0], left_u[n])
        ghost_right[n] = extrap(u[n, -2], u[n, -1], right_u[n])

        # Update interior points using finite differences
        u[n+1, 0] = 2*u[n, 0] - u[n-1, 0] + (dt**2 / dx**2) * (u[n, 1] - 2*u[n, 0] + ghost_left[n])
        for i in range(1, Nx - 1):
            u[n+1, i] = 2*u[n, i] - u[n-1, i] + (dt**2 / dx**2) * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
        u[n+1, -1] = 2*u[n, -1] - u[n-1, -1] + (dt**2 / dx**2) * (ghost_right[n] - 2*u[n, -1] + u[n, -2])

    return u, get_uut(domain, u0, u)
    
u_fd = None
ut_fd = None


def main():
    args = parse_args()
    odil.setup_outdir(args)
    problem, state = make_problem(args)
    global u_fd
    global ut_fd
    # u_fd, ut_fd = solve_fd_strong_dirichlet(problem.domain)
    u_fd, ut_fd = solve_fd_ghosts_cells(problem.domain)
    # print(u_fd.shape, ut_fd.shape, "fd")
    callback = odil.make_callback(
        problem, args, plot_func=plot_func, history_func=history_func, report_func=report_func
    )
    odil.util.optimize(args, args.optimizer, problem, state, callback)
    # odil.util.optimize(args, args.optimizer, problem, state, callback, factr=10000)

    with open("done", "w"):
        pass


if __name__ == "__main__":
    main()
