import os
import numpy as np
import pretty_midi
import librosa
import soundfile as sf

def midi_to_audio(midi_file):
    """Convert MIDI to audio using pretty_midi."""
    pm = pretty_midi.PrettyMIDI(midi_file)
    # Set instrument to Acoustic Grand Piano (Program Number 0)
    for instrument in pm.instruments:
        instrument.program = 0
    audio = pm.synthesize(fs=44100)
    return audio, 44100

def add_noise_in_frequency_domain(signal, noise_level):
    """
    Add noise to the signal in the frequency domain to perturb the audio signal.
    At 100% noise level, the signal becomes entirely random.
    """
    # Perform STFT
    n_fft = 2048
    hop_length = n_fft // 4
    D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    
    # Get magnitude and phase
    magnitude, phase = np.abs(D), np.angle(D)
    
    # Generate random noise for magnitude and phase
    random_magnitude = np.random.rand(*magnitude.shape) * np.max(magnitude)
    random_phase = np.random.uniform(-np.pi, np.pi, size=phase.shape)
    
    # Interpolate between original and random components based on noise level
    noisy_magnitude = (1 - noise_level) * magnitude + noise_level * random_magnitude
    noisy_phase = (1 - noise_level) * phase + noise_level * random_phase
    
    # Reconstruct the noisy signal
    D_noisy = noisy_magnitude * np.exp(1j * noisy_phase)
    noisy_signal = librosa.istft(D_noisy, hop_length=hop_length, length=len(signal))
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(noisy_signal))
    if max_val > 0:
        noisy_signal = noisy_signal / max_val
    
    return noisy_signal

def add_phase_jitter(signal, noise_level):
    """
    Add phase jitter to the signal in the frequency domain.
    noise_level ranges from 0.0 (no noise) to 1.0 (maximum noise).
    """
    # Perform STFT
    n_fft = 2048
    hop_length = n_fft // 4
    D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    
    # Get magnitude and phase
    magnitude, phase = np.abs(D), np.angle(D)
    
    # Generate random phase noise
    max_phase_noise = np.pi * 2 * noise_level  # Max phase noise scales with noise_level
    phase_noise = np.random.uniform(-max_phase_noise, max_phase_noise, size=phase.shape)
    
    # Add phase noise
    noisy_phase = phase + phase_noise
    
    # Reconstruct the signal
    D_noisy = magnitude * np.exp(1j * noisy_phase)
    noisy_signal = librosa.istft(D_noisy, hop_length=hop_length, length=len(signal))
    
    # Normalize the signal
    max_val = np.max(np.abs(noisy_signal))
    if max_val > 0:
        noisy_signal = noisy_signal / max_val
    
    return noisy_signal

def add_time_domain_jitter(signal, noise_level, sr):
    """
    Introduce random time shifts (jitter) to small frames of the signal.
    """
    frame_length_ms = 10  # Frame length in milliseconds
    frame_length_samples = int(sr * frame_length_ms / 1000)
    
    max_shift_ms = 10 * noise_level  # Maximum shift in milliseconds
    max_shift_samples = int(sr * max_shift_ms / 1000)
    
    num_frames = len(signal) // frame_length_samples
    output_signal = np.zeros_like(signal)
    
    for i in range(num_frames):
        start = i * frame_length_samples
        end = start + frame_length_samples
        frame = signal[start:end]
        
        # Random shift within the maximum shift range
        shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
        shifted_start = start + shift
        shifted_end = shifted_start + len(frame)
        
        # Ensure indices are within bounds
        if shifted_start < 0:
            frame = frame[-shifted_start:]
            shifted_start = 0
        if shifted_end > len(signal):
            frame = frame[:len(signal) - shifted_start]
            shifted_end = len(signal)
        
        # Overlap-add the shifted frame
        output_signal[shifted_start:shifted_end] += frame
    
    # Normalize the signal
    max_val = np.max(np.abs(output_signal))
    if max_val > 0:
        output_signal = output_signal / max_val
    
    return output_signal


def process_midi_files(data_dir, output_dir, noise_levels):
    """Process all MIDI files in the dataset."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    midi_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(root, file))
    
    print(f"Found {len(midi_files)} MIDI files.")
    
    for midi_file in midi_files:
        try:
            print(f"Processing: {midi_file}")
            base_name = os.path.splitext(os.path.basename(midi_file))[0]
            
            # Generate audio from MIDI
            y, sr = midi_to_audio(midi_file)
            
            # Normalize the audio
            y = y / np.max(np.abs(y))
            
            # Create noisy versions
            for noise_percent in noise_levels:
                noise_level = noise_percent / 100.0  # Convert to 0.0 - 1.0
                
                if noise_level == 0:
                    # No noise added; use original signal
                    y_noisy = y
                else:
                    # Add phase jitter
                    # y_noisy = add_phase_jitter(y, noise_level)
                    y_noisy = add_time_domain_jitter(y, noise_level, sr)
                
                # Normalize to prevent clipping
                y_noisy = y_noisy / np.max(np.abs(y_noisy))
                
                # Save file
                output_file = f"{base_name}_noise_{int(noise_percent)}percent.wav"
                output_path = os.path.join(output_dir, output_file)
                sf.write(output_path, y_noisy, sr)
                print(f"Saved: {output_path}")
                    
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")


# Main execution
if __name__ == "__main__":
    np.random.seed(42)
    data_directory = 'EMOPIA_1.0/midis'
    output_directory = 'EMOPIA_1.0_noisy_time'
    noise_levels = [0, 25, 50, 75, 100]  # Adjust as needed
    
    process_midi_files(data_directory, output_directory, noise_levels)
