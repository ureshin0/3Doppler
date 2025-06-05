# _generate.py
import healjax as hp
import jax
import jax.numpy as jnp
from functools import partial
from spectrum import incline, doppler_shift
from scipy.stats import norm
import matplotlib.pyplot as plt
from map import map5

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

def matrix(nside, vrot, inclination, phase, wavelengths, line_profile, output_resolution):
    npix = hp.nside2npix(nside)
    vmap = jax.vmap(partial(hp.pix2ang, 'ring', nside))
    theta0, phi0 = vmap(jnp.arange(npix))
    phi0 = phi0 + phase * 2 * jnp.pi
    vlos = vrot * jnp.cos(jnp.pi/2 - inclination) * jnp.sin(theta0) * jnp.sin(phi0)
    theta, phi = incline(theta0, phi0, jnp.pi/2 - inclination)

    wl_min = wavelengths[0] * doppler_shift(-vrot)
    wl_max = wavelengths[-1] * doppler_shift(vrot)
    observed_wl = jnp.linspace(wl_min, wl_max, output_resolution)

    visible = ((-jnp.pi/2 < phi) & (phi < jnp.pi/2)).astype(float)
    shifted = wavelengths[:, None] * doppler_shift(vlos)
    profile_ext = line_profile[:, None] * jnp.ones(npix)
    interp = jax.vmap(jnp.interp, in_axes=(None, 0, 0))
    interped = interp(observed_wl, shifted.T, profile_ext.T).T
    M = visible * interped * jnp.sin(theta) * jnp.cos(phi)
    return M, observed_wl

#%%
def matrices(nside, vrot, inclination, n_phase, wavelengths, line_profile, output_resolution, w):
    _, obs_wl = matrix(nside, vrot, inclination, 0, wavelengths, line_profile, output_resolution)
    phases = jnp.arange(n_phase) / n_phase

    def scaled(phase, wi):
        M, _ = matrix(nside, vrot, inclination, phase, wavelengths, line_profile, output_resolution)
        return wi * M

    M_stack = jax.vmap(scaled)(phases, w)
    W = M_stack.reshape((n_phase * output_resolution, -1))
    return W, obs_wl

# 合成データ作成
if __name__ == "__main__":
    def generate(true_map, true_line_profile, true_w, true_i, true_v, n_phase=8, out_res=100, key=0):
        if len(true_w) != n_phase:
            print("len(true_w) != n_phase")
            return

        @jax.jit
        def make_W(i, w, v):
            return matrices(8, v, i, n_phase, wl, line_profile, out_res, w)[0]
        true_W = make_W(true_i, true_w, true_v)

        d = true_W @ true_map

        # ノイズ追加
        sigma = 0.01 * jnp.max(jnp.abs(d))
        key = jax.random.PRNGKey(key)
        d_noisy = d + jax.random.normal(key, d.shape) * sigma

        return d_noisy

    wl0 = 656.28
    wl = jnp.linspace(wl0 - 0.05, wl0 + 0.05, 100)
    nu0 = 1e7 / wl0
    nu = 1e7 / wl
    line_profile = 1 - 0.8 * norm.pdf(nu, nu0, 0.3) / jnp.max(norm.pdf(nu, nu0, 0.3))

    n_phase = 8
    out_res = 100

    true_w = jnp.array([float(k+1) for k in range(n_phase)])
    true_i = jnp.deg2rad(jnp.array(range(0, 91, 10)))
    true_v = jnp.array(range(20,61,20))

    d = jnp.zeros((len(true_i), len(true_v), n_phase*out_res))
    for m, i in enumerate(true_i):
        for n, v in enumerate(true_v):
            # d[m][n] = generate(map5, line_profile, true_w, i, v)
            d = d.at[m, n].set(generate(map5, line_profile, true_w, i, v))

    jnp.save('true_i.npy', jnp.rad2deg(true_i))
    jnp.save('true_v.npy', true_v)
    jnp.save('d.npy', d)
    print("saved npy files!")