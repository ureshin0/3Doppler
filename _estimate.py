# # _estimate.py
import healjax as hp
import jax
import jax.numpy as jnp
from functools import partial
from spectrum import incline, doppler_shift
import optax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.stats import norm
import matplotlib.pyplot as plt

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

def matrices(nside, vrot, inclination, n_phase, wavelengths, line_profile, output_resolution, w):
    _, obs_wl = matrix(nside, vrot, inclination, 0, wavelengths, line_profile, output_resolution)
    phases = jnp.arange(n_phase) / n_phase

    def scaled(phase, wi):
        M, _ = matrix(nside, vrot, inclination, phase, wavelengths, line_profile, output_resolution)
        return wi * M

    M_stack = jax.vmap(scaled)(phases, w)
    W = M_stack.reshape((n_phase * output_resolution, -1))
    return W, obs_wl

if __name__ == "__main__":
    def estimate(d, true_line_profile, n_phase=8, out_res=100, key=0):
        @jax.jit
        def make_W(i, w, v):
            return matrices(8, v, i, n_phase, wl, true_line_profile, out_res, w)[0]

        #=== w の最適化: i, v は事前分布からサンプリングして周辺尤度を最大化 ===
        # 事前から i, v のサンプルを事前に取得
        N_samps = 16
        key = jax.random.PRNGKey(key)
        u_samps = jax.random.uniform(key, (N_samps,), minval=-1.0, maxval=1.0)
        i_samps = jnp.arccos(u_samps)
        key, subkey = jax.random.split(key)
        v_samps = jax.random.uniform(subkey, (N_samps,), minval=0.0, maxval=50.0)

        def loss_w(w, d):
            # i_samps, v_samps に対して平均負対数周辺尤度
            def single_loss(i, v):
                Wiv = make_W(i, w, v)
                C = Sigma_d + Wiv @ Sigma_a @ Wiv.T
                sign, logdet = jnp.linalg.slogdet(C)
                invC_d = jnp.linalg.solve(C, d)
                return 0.5 * (d @ invC_d + logdet)
            losses = jax.vmap(single_loss)(i_samps, v_samps)
            return jnp.mean(losses)

        def optimize_w(d, lr=1e-2, iters=300):
            w = jnp.ones(n_phase)
            opt = optax.adam(lr)
            state = opt.init(w)

            @jax.jit
            def step(w, state, d):
                loss_val, grad = jax.value_and_grad(loss_w)(w, d)
                updates, state = opt.update(grad, state, w)
                w = optax.apply_updates(w, updates)
                return w, state

            for _ in range(iters):
                w, state = step(w, state, d)
            return w

        # 傾斜角iと回転速度vをベイズ推定
        def model_iv(data, w_fixed):
            u = numpyro.sample('u', dist.Uniform(-1, 1))
            i = jnp.arccos(u)
            v = numpyro.sample('v', dist.Uniform(0.0, 50.0))
            Wiv = make_W(i, w_fixed, v)
            C = Sigma_d + Wiv @ Sigma_a @ Wiv.T
            numpyro.sample('obs', dist.MultivariateNormal(loc=jnp.zeros(data.shape[0]), covariance_matrix=C), obs=data)

        # まずwを最適化
        w_est = optimize_w(d)

        # 次にi,vのMCMC推定
        kernel = NUTS(lambda data: model_iv(data, w_est))
        mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
        mcmc.run(key, d)
        samples = mcmc.get_samples()
        i_s = jnp.rad2deg(jnp.arccos(samples['u']))
        v_s = samples['v']

        return w_est, i_s, v_s
    
    wl0 = 656.28
    wl = jnp.linspace(wl0 - 0.05, wl0 + 0.05, 100)
    nu0 = 1e7 / wl0
    nu = 1e7 / wl
    line_profile = 1 - 0.8 * norm.pdf(nu, nu0, 0.3) / jnp.max(norm.pdf(nu, nu0, 0.3))

    n_phase = 8
    out_res = 100
    
    Sigma_d = 0.01 * jnp.eye(n_phase * out_res)
    Sigma_a = 0.01 * jnp.eye(hp.nside2npix(8))

    # データ読み込み
    true_i = jnp.load('true_i.npy')
    true_v = jnp.load('true_v.npy')
    d = jnp.load('d.npy')
    
    for n in range(len(true_v)):
        mean_i = []
        accute_i = []
        median_i = []
        lower_i = []
        upper_i = []
        for m in range(len(true_i)):
            w_est, i_s, v_s = estimate(d[m][n], line_profile, n_phase=8, out_res=100, key=0)
            print("Estimated w:", w_est)
            print("true v [km/s]:", true_v[n], "/ Estimated v:", jnp.mean(v_s))
            print(f"true i [deg]: {true_i[m]} / Estimated i: {jnp.mean(i_s)} / 90% CI: [{jnp.percentile(i_s, 5)}, {jnp.percentile(i_s, 95)}]")
            mean_i.append(jnp.mean(i_s))
            accute_i.append(90 - max(0, (jnp.mean(i_s))-90))
            median_i.append(jnp.percentile(i_s, 50))
            lower_i.append(jnp.percentile(i_s, 5))
            upper_i.append(jnp.percentile(i_s, 95))
        mean_i = jnp.array(mean_i)
        accute_i = jnp.array(accute_i)
        median_i = jnp.array(median_i)
        lower_i = jnp.array(lower_i)
        upper_i = jnp.array(upper_i)
        plt.errorbar(jnp.array(true_i), median_i, yerr=[median_i-lower_i, upper_i-median_i], fmt='o', label="90% CI")
        plt.scatter(true_i, mean_i, color='red')
        plt.scatter(true_i, accute_i, color='red', label="mean")
        plt.plot(true_i, true_i, '--', label = "true", color='orange')
        plt.plot(true_i, 180-true_i, '--', label = "true", color='orange')
        plt.xlim([0,90])
        plt.ylim([0,180])
        plt.xlabel("true inclination[deg]")
        plt.ylabel("estimated inclination[deg]")
        plt.legend()
        filename = f'{true_v[n]}kms_kaidan.png'
        plt.savefig(filename)
        plt.clf()
        print(f"----------", filename, "----------")