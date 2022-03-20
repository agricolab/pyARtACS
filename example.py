from artacs import CombKernel
import numpy as np
import time

fs = 1000
duration = 1
t = np.linspace(0, duration, duration * fs)
data = np.atleast_2d(np.sin(2 * 10 * np.pi * t))

ck = CombKernel(
    freq=10, fs=1000, width=1, left_mode="uniform", right_mode="none"
)

T = []
for i in range(10_000):
    t0 = time.time_ns()
    out = ck.apply(data)
    T.append((time.time_ns() - t0) / 1000)
print(
    f"99% CI of calculation time for 1s data is between {np.percentile(T, 0.5):3.2f} to  {np.percentile(T, 99.5):3.2f} µs"
)
import matplotlib.pyplot as plt

plt.plot(data.T, label="Raw")
plt.plot(out.T, label="Filtered")
plt.legend()
plt.show()

