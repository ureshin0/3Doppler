import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from const import c0

# 理論計算
sigma = 0.3/10**7
c = c0 * 10**(-3)  # 光速(km/s)
v = 20      # 速度(km/s)
λ_0 = 656.28  # 中心波長(nm)

def integrand(θ, φ, λ):
    factor1 = 1 / (sigma * np.sqrt(2 * np.pi))
    factor2 = np.sqrt((c - v * np.sin(θ) * np.sin(φ)) / (c + v * np.sin(θ) * np.sin(φ)))
    exp_part = np.exp(-(1/2) * ((1/factor2 * 1/λ - 1/λ_0) / sigma) ** 2)
    return (1-0.8*exp_part) * np.sin(θ) * np.cos(φ) * np.sin(θ) # 最後のsinθはヤコビアンのやつ

λ_values = np.linspace(656.190, 656.370, 200)
integral_values = []

# θ=0~π, φ=-π/2~π/2で積分
for λ in λ_values:
    result, _ = dblquad(integrand, -np.pi/2, np.pi/2, lambda φ: 0, lambda φ: np.pi, args=(λ,))
    integral_values.append(result)

integral_values /= np.max(integral_values)
plt.plot(λ_values, integral_values, label="theoretical")


# HealPIXを使ったobserve
import spectrum
import map
from scipy.stats import norm
map = map.make_spot(0, 0, 0, 1)
vrot = 20  # 回転速度 (km/s)
wl0=656.28 #(nm)
wavelengths = np.linspace(wl0-0.05, wl0+0.05, 1000)
nu0=10**7/wl0 #(/cm)
nu=10**7/wavelengths
line_profile = 1-0.8*norm.pdf(nu, nu0, 0.3)/np.max(norm.pdf(nu, nu0, 0.3))
observed_wavelengths, observed_spectrum = spectrum.observe_spectrum(map, vrot, wavelengths, line_profile, normalize=True)
plt.plot(observed_wavelengths, observed_spectrum, "--", label="HEALPix")

# 元の吸収線プロファイル
plt.plot(wavelengths,line_profile/np.max(line_profile), label="line profile")

plt.xlabel('Wavelength (nm)')
plt.ylabel('Flux (normalized)')
plt.legend()
plt.grid(True)
plt.savefig("theoretical.png")
