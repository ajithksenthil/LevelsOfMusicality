import os
import numpy as np
import pretty_midi
import librosa
import soundfile as sf
import colorednoise

def midi_to_audio(midi_file):
    """Convert MIDI to audio using pretty_midi."""
    pm = pretty_midi.PrettyMIDI(midi_file)
    audio = pm.synthesize(fs=44100)
    return audio, 44100

def calculate_power(signal):
    """Calculate the power of a signal."""
    return np.mean(signal ** 2)

def add_noise_with_snr(signal, desired_snr_db):
    """
    Add noise to signal based on desired Signal-to-Noise Ratio (SNR) in dB.
    A lower SNR means more noise and more distortion.
    """
    # Calculate signal power
    signal_power = calculate_power(signal)
    
    # Generate pink noise (exponent=1 for pink noise)
    noise = colorednoise.powerlaw_psd_gaussian(exponent=1.0, size=len(signal))
    
    # Calculate desired noise power based on SNR
    desired_noise_power = signal_power / (10 ** (desired_snr_db / 10))
    
    # Scale noise to achieve desired SNR
    current_noise_power = calculate_power(noise)
    scaling_factor = np.sqrt(desired_noise_power / current_noise_power)
    scaled_noise = noise * scaling_factor
    
    # Add noise to signal
    noisy_signal = signal + scaled_noise
    
    # Normalize to prevent clipping
    max_val = max(abs(noisy_signal))
    if max_val > 1:
        noisy_signal = noisy_signal / max_val
        
    return noisy_signal

def process_midi_files(data_dir, output_dir, noise_levels):
    """Process all MIDI files in the dataset."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert noise percentages to SNR values
    # Higher noise % = Lower SNR
    # 100% noise → SNR = -20 dB (very noisy)
    # 0% noise → SNR = 40 dB (very clean)
    max_snr = 40  # dB
    min_snr = -20  # dB
    
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
            
            # Create noisy versions
            for noise_percent in noise_levels:
                # Convert noise percentage to SNR
                # As noise_percent increases, SNR decreases
                snr_db = max_snr - (noise_percent/100) * (max_snr - min_snr)
                
                # Add noise based on SNR
                y_noisy = add_noise_with_snr(y, snr_db)
                
                # Save file
                output_file = f"{base_name}_noise_{int(noise_percent)}percent.wav"
                output_path = os.path.join(output_dir, output_file)
                sf.write(output_path, y_noisy, sr)
                print(f"Saved: {output_path} (SNR: {snr_db:.1f} dB)")
                
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")

# Main execution
if __name__ == "__main__":
    data_directory = 'EMOPIA_1.0/midis'
    output_directory = 'EMOPIA_1.0_noisy'
    noise_levels = np.arange(10, 110, 10)  # 10% to 100% noise levels
    
    process_midi_files(data_directory, output_directory, noise_levels)