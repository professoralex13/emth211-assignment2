import numpy as np
import matplotlib.pyplot as plt

def log_fit(x, y):
    A = np.column_stack([np.ones_like(x), x])
    Q, R = np.linalg.qr(A)

    return np.linalg.solve(R, Q.T @ np.log(y))

year = np.arange(1993, 2024 + 1, 1)

flops_per_second = np.array([
    1.240e11, 1.700e11, 1.700e11, 3.680e11, 1.300e12, 1.300e12 ,2.400e12, 4.900e12,
    7.200e12, 3.590e13, 3.590e13, 7.070e13, 2.806e14, 2.806e14 ,4.782e14, 1.100e15,
    1.800e15, 2.600e15, 1.050e16, 1.760e16, 3.390e16, 3.390e16 ,3.390e16, 9.300e16,
    9.300e16, 1.435e17, 1.486e17, 4.420e17, 4.420e17, 1.102e18 ,1.194e18, 1.742e18
])


fig1, ax1 = plt.subplots()

ax1.semilogy()
ax1.set_ylabel("Flops per Second")
ax1.set_xlabel("Year")


# Part A
ax1.plot(year, flops_per_second, label="Source data")
fig1.savefig('1a.graph.png')

# Part B

ln_a, b = log_fit(year, flops_per_second)

ax1.plot(year, np.exp(ln_a + b * year), label="Logarithmic line of best fit")
fig1.legend()
fig1.savefig('1b.graph.png')

# Part C

[[i]] = np.where(year == 2015)

ln_a1, b1 = log_fit(year[:i + 1], flops_per_second[:i + 1])
ln_a2, b2 = log_fit(year[i:], flops_per_second[i:])

fig2, ax2 = plt.subplots()

ax2.semilogy()
ax2.set_ylabel("Flops per Second")
ax2.set_xlabel("Year")

ax2.plot(year, flops_per_second, label="Source data")
ax2.plot(year[:i + 1], np.exp(ln_a1 + b1 * year[:i + 1]), label="Pre 2015 Logarithmic line of best fit 1")
ax2.plot(year[i:], np.exp(ln_a2 + b2 * year[i:]), label="Post 2015 Logarithmic line of best fit")

fig2.legend()

fig2.savefig('1c.graph.png')

# Part D

max_error = max(np.abs(np.log(flops_per_second) - (ln_a + b * year)))

actual_1969 = 3.6e7
pred_1969 = np.exp(ln_a + b * 1969)

print(f"Maximum Error: {max_error:.2f}")
print(f"Predicted Flops in 1969: {pred_1969:.0f}")
print(f"Extrapolation Error (1969): {np.log(actual_1969) - np.log(pred_1969):.2f}")

