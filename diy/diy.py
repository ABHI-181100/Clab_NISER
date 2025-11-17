import numpy as np
import matplotlib.pyplot as plt

# ------------------- SVD function -------------------
def SVD(A):
    """Perform Singular Value Decomposition."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U, S, Vt


# ------------------- Variance explained -------------------
def variance(S):
    """Compute normalized variance (energy) of singular values."""
    return np.round(S**2 / np.sum(S**2), 6)


# ------------------- Denoising function -------------------
def svd_denoise(A, rank):
    """
    Perform neural signal denoising by keeping top `rank` singular components.

    Args:
        A (array): Neural signal matrix (channels × time)
        rank (int): Number of dominant components to retain

    Returns:
        array: Denoised signal matrix
    """
    U, S, Vt = SVD(A)
    
    # Truncate singular values beyond 'rank'
    U_k = U[:, :rank]
    S_k = np.diag(S[:rank])
    Vt_k = Vt[:rank, :]
    
    # Reconstruct the denoised signal
    A_denoised = U_k @ S_k @ Vt_k
    return A_denoised


# ------------------- Example Simulation -------------------
# Create a noisy neural-like signal
np.random.seed(0)
t = np.linspace(0, 1, 1000)
signal = np.array([
    np.sin(2 * np.pi * 10 * t),
    np.sin(2 * np.pi * 10 * t + np.pi/4),
    np.sin(2 * np.pi * 20 * t),
    np.sign(np.sin(0.5*t)),
    np.sin(0.5*t + 0.5) * np.exp(-0.001*t) 
])
# Add Gaussian noise
noisy_signal = signal + 0.3 * np.random.randn(*signal.shape)


plt.subplot(3, 1, 1)
plt.title("Original Neural Signals")
plt.plot(t, signal.T)
plt.ylabel("Amplitude")
plt.show()
# ------------------- Plotting -------------------

U, S, Vt = SVD(noisy_signal)
y = np.cumsum(variance(S))*100


plt.plot(np.cumsum(variance(S))*100)
plt.title("Cumulative variance explained")
plt.xlabel("Number of singular values")
plt.ylabel("Variance (%)")
plt.show()

# Denoise with SVD
rank = 3
denoised_signal = svd_denoise(noisy_signal, rank)



plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title("Noisy Neural Signals")
plt.plot(t, noisy_signal.T)
plt.subplot(2, 1, 2)
plt.title(f"Denoised Neural Signals (Rank={rank})")
plt.plot(t, denoised_signal.T)
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()

from scipy.signal import welch

f, Pxx_noisy = welch(noisy_signal[0], fs=1000)
f, Pxx_denoised = welch(denoised_signal[0], fs=1000)

plt.semilogy(f, Pxx_noisy, label='Noisy')
plt.semilogy(f, Pxx_denoised, label='Denoised')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density')
plt.legend()
plt.title('Power spectrum before vs after SVD denoising')
plt.show()
