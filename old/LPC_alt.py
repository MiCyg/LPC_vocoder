import librosa
import numpy as np
from scipy.linalg import toeplitz, solve_toeplitz

# TO JEST NIESPRAWDZONY KOD Podjumany z internetu

def calculate_lpc_coefficients(signal, order=10):
    # Step 1: Frame the signal
    frame_size = 512
    frames = librosa.util.frame(signal, frame_length=frame_size, hop_length=frame_size // 2)

    # Step 2: Windowing
    frames *= np.hamming(frame_size)

    # Step 3: Autocorrelation Calculation
    autocorr = np.correlate(frames[0, :], frames[0, :], mode='full')
    autocorr = autocorr[frame_size - 1:]

    # Step 4: Levinson-Durbin Algorithm
    toeplitz_matrix = toeplitz(autocorr[:order])
    rhs_vector = -autocorr[1:order + 1]
    lpc_coefficients = solve_toeplitz((toeplitz_matrix, rhs_vector), sym=True)

    return lpc_coefficients

def encode_decode_with_lpc(input_file, output_file, lpc_order=10):
    # Load the input audio file
    input_signal, sr = librosa.load(input_file, sr=None)

    # Calculate LPC coefficients
    lpc_coefficients = calculate_lpc_coefficients(input_signal, order=lpc_order)

    # Encode the signal using LPC
    encoded_signal = np.convolve(input_signal, lpc_coefficients, mode='full')

    # Decode the signal using LPC
    decoded_signal = np.convolve(encoded_signal, lpc_coefficients[::-1], mode='full')

    # Save the decoded signal to a new WAV file
    librosa.output.write_wav(output_file, decoded_signal, sr)

if __name__ == "__main__":
    input_wav_file = "odbiemrz.wav"
    output_wav_file = "output_lpc.wav"
    lpc_order = 10

    encode_decode_with_lpc(input_wav_file, output_wav_file, lpc_order)
    print(f"Encoding and decoding with LPC completed. Output saved to {output_wav_file}")