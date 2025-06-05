
import healjax as hp

from const import c0

import time
import jax
import numpy as np
import jax.numpy as jnp
from functools import partial

def incline(theta0, phi0, alpha):
    # Convert spherical to Cartesian
    x0 = jnp.sin(theta0) * jnp.cos(phi0)
    y0 = jnp.sin(theta0) * jnp.sin(phi0)
    z0 = jnp.cos(theta0)

    # Apply rotation matrix around the y-axis
    x = jnp.cos(alpha) * x0 + jnp.sin(alpha) * z0
    y = y0
    z = -jnp.sin(alpha) * x0 + jnp.cos(alpha) * z0

    # Convert back to spherical
    theta = jnp.arccos(z)
    phi = jnp.arctan2(y, x)

    return theta, phi

def limb_darkening(u, mu):
    return 1-u*(1-mu)

def doppler_shift(vlos):
    """
    視線速度からドップラーシフトの倍率を計算する。

    Parameters
    ----------
    v : float
        視線速度(km/s)。視線から遠ざかる向きが正。
    """
    c = c0 * 10**(-3)  # 光速 (km/s)
    beta = vlos/c
    return (1 + beta) / jnp.sqrt(1 - beta**2)

def observe_spectrum(map, vrot, wavelengths, line_profile, inclination=jnp.pi/2, phase=0., u=0, output_resolution=1000 ,normalize=True):
    """
    マップを元に、ドップラー効果を考慮したスペクトルを生成する。

    Parameters
    ----------
    map : array
        星表面のマップ。
    vrot : float
        星の自転速度(km/s)。
    wavelengths : array
        一様なスペクトルのシフト前の波長(nm)。
    line_profile : array
        一様なスペクトル強度。
    inclination : float
        自転軸の傾き(rad)。pi/2のとき視線に垂直。
    phase : float
        観測するフェーズ。1.0で1自転。
    u : float
        周縁減光の強さを表す係数。0から1まで。
    output_resolution : int
        出力スペクトルの解像度。
    normal : boolean
        最後に正規化するかどうか。
    """
    start=time.time()

    nside = hp.npix2nside(len(map))
    vmap_pix2ang = jax.vmap(partial(hp.pix2ang, 'ring', nside))
    theta0, phi0 = vmap_pix2ang(jnp.arange(len(map)))

    time1=time.time()
#    print(f"time1 = {time1-start}")

    # 自転
    phi0 += phase*2*jnp.pi

    # 観測者からの視線速度を計算
    vlos = vrot * jnp.cos(jnp.pi/2-inclination) * jnp.sin(theta0) * jnp.sin(phi0)

    # 観測者から見た座標に変換
    theta, phi = incline(theta0, phi0, jnp.pi/2-inclination)

    # スペクトル波長範囲を設定
    wl_min = wavelengths[0] * doppler_shift(-vrot)
    wl_max = wavelengths[-1] * doppler_shift(vrot)
    observed_wavelengths = jnp.linspace(wl_min, wl_max, output_resolution)

    time2=time.time()
#    print(f"time2 = {time2-time1}")

    # 各ピクセルについて放射をドップラーシフトさせ、スペクトルに足し合わせる
    pixel_intensity = jnp.where((-jnp.pi / 2 < phi) & (phi < jnp.pi / 2), map, 0)
    shifted_wavelengths = wavelengths[:,jnp.newaxis] * doppler_shift(vlos)
    mu = jnp.sin(theta) * jnp.cos(phi)
    line_profile = line_profile[:,jnp.newaxis] * jnp.ones_like(vlos)
    vmap_interp = jax.vmap(jnp.interp, in_axes=(None,0,0))
    interped = vmap_interp(observed_wavelengths, shifted_wavelengths.T, line_profile.T).T
    observed_spectrum = jnp.sum(pixel_intensity * interped * limb_darkening(u,mu) * jnp.sin(theta) * jnp.cos(phi), axis=1)

    # スペクトルの正規化
    if normalize:
        observed_spectrum /= jnp.max(observed_spectrum)

    time3=time.time()
#    print(f"time3 = {time3-time2}")
#    print(f"time = {time3-start}")

    return observed_wavelengths, observed_spectrum


if __name__ == "__main__":
    import jax
    jax.config.update("jax_enable_x64", True)

    from scipy.stats import norm
    from map import map1, map2, map4, map5
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    vrot = 20  # 回転速度 (km/s)
    wl0=656.28
    wl = jnp.linspace(wl0-0.05, wl0+0.05, 1000)
    nu0=10**7/wl0 #(/cm)
    nu=10**7/wl
    line_profile = 1-0.8*norm.pdf(nu, nu0, 0.3)/jnp.max(norm.pdf(nu, nu0, 0.3))
    for i in range(8):
        wavelengths, spectrum = observe_spectrum(map5, vrot, wl, line_profile, inclination=jnp.pi/2, phase=0.125*i, normalize=True)
        plt.plot(wavelengths, spectrum, label=f"t={i}T/8", color=cm.hot(i/8), zorder=8-i)
    wavelengths, spectrum = observe_spectrum(map2, vrot, wl, line_profile, inclination=jnp.pi/2, phase=0.125*1, normalize=True)
    plt.xlabel("wavelengths [nm]")
    plt.ylabel("normalized flux")
    plt.legend()
    plt.title("Observed Spectra at Different Rotational Phases")

    plt.savefig("./wai2.png")
    plt.clf()
    print(len(wavelengths))
    print(len(spectrum))
    print("saved")