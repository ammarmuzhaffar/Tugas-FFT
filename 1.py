
import numpy as np
import matplotlib.pyplot as plt

# Create a square wave function in the range -A/2 < T < A/2
A = 10
T = 2 * A
t = np.linspace(-A/2, A/2, 1000)
square_wave = np.where((t >= -A/4) & (t <= A/4), 1, 0)

# Plot the square wave function
plt.figure()
plt.plot(t, square_wave)
plt.xlabel('Time (T)')
plt.ylabel('Amplitude')
plt.title('Square Wave Function (-A/2 < T < A/2)')
plt.grid()
plt.show()

# Perform 1D FFT for -A/2 < T < A/2
fft_result = np.fft.fft(square_wave)
frequencies = np.fft.fftfreq(len(fft_result))

# Plot the magnitude spectrum
plt.figure()
plt.plot(frequencies, np.abs(fft_result))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('1D FFT Magnitude Spectrum (-A/2 < T < A/2)')
plt.grid()
plt.show()

# Create a 2D square wave image for -A/2 < T < A/2
t_2D = np.linspace(-A/2, A/2, 1000)
square_wave_2D = np.outer(np.where((t_2D >= -A/4) & (t_2D <= A/4), 1, 0), np.where((t_2D >= -A/4) & (t_2D <= A/4), 1, 0))

# Perform 2D FFT for -A/2 < T < A/2
fft_result_2D = np.fft.fft2(square_wave_2D)
fft_result_2D = np.fft.fftshift(fft_result_2D)

# Plot the magnitude spectrum
plt.figure()
plt.imshow(np.abs(fft_result_2D), cmap='viridis', extent=(t_2D.min(), t_2D.max(), t_2D.min(), t_2D.max()))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D FFT Magnitude Spectrum (-A/2 < T < A/2)')
plt.colorbar()
plt.show()

import librosa
import librosa.display
import matplotlib.pyplot as plt


audio_file = librosa.example('Sample')
y, sr = librosa.load(audio_file)

# Compute MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # You can adjust the number of coefficients

# Plot MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(mfccs, ref=np.max), y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC')
plt.show()




