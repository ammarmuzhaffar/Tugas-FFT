
import numpy as np
import matplotlib.pyplot as plt

# Create a square wave function in the range -A < T < A
A = 10
T = 2 * A
t = np.linspace(-A, A, 1000)
square_wave = np.where((t >= -A/2) & (t <= A/2), 1, 0)

# Plot the square wave function
plt.figure()
plt.plot(t, square_wave)
plt.xlabel('Time (T)')
plt.ylabel('Amplitude')
plt.title('Square Wave Function (-A < T < A)')
plt.grid()
plt.show()

# Perform 1D FFT for -A < T < A
fft_result = np.fft.fft(square_wave)
frequencies = np.fft.fftfreq(len(fft_result))

# Plot the magnitude spectrum
plt.figure()
plt.plot(frequencies, np.abs(fft_result))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('1D FFT Magnitude Spectrum (-A < T < A)')
plt.grid()
plt.show()

# Create a 2D square wave image for -A < T < A
t_2D = np.linspace(-A, A, 1000)
square_wave_2D = np.outer(np.where((t_2D >= -A/2) & (t_2D <= A/2), 1, 0), np.where((t_2D >= -A/2) & (t_2D <= A/2), 1, 0))

# Perform 2D FFT for -A < T < A
fft_result_2D = np.fft.fft2(square_wave_2D)
fft_result_2D = np.fft.fftshift(fft_result_2D)

# Plot the magnitude spectrum
plt.figure()
plt.imshow(np.abs(fft_result_2D), cmap='viridis', extent=(t_2D.min(), t_2D.max(), t_2D.min(), t_2D.max()))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D FFT Magnitude Spectrum (-A < T < A)')
plt.colorbar()
plt.show()

import librosa
import librosa.display

square_wave_audio = square_wave.astype(np.float32)

# Compute MFCCs for -A < T < A
mfccs = librosa.feature.mfcc(y=square_wave_audio, sr=1000, n_mfcc=13)  # Adjust sr as needed

# Plot MFCC
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC (-A < T < A)')
plt.show()

