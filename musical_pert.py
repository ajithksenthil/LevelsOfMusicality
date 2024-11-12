import os
import numpy as np
import copy  # Import the copy module
import pretty_midi
import soundfile as sf

def midi_to_audio(pm, fs=44100):
    """Convert PrettyMIDI object to audio using the built-in synthesizer."""
    audio = pm.synthesize(fs=fs)
    return audio, fs

def randomize_midi_notes(pm, noise_level):
    """
    Randomly alter the pitches of notes in the MIDI data based on noise_level.
    noise_level ranges from 0.0 (no changes) to 1.0 (all notes randomized).
    """
    for instrument in pm.instruments:
        num_notes = len(instrument.notes)
        num_notes_to_randomize = int(noise_level * num_notes)
        
        # Randomly select notes to randomize
        indices = np.random.choice(num_notes, num_notes_to_randomize, replace=False)
        
        for idx in indices:
            original_note = instrument.notes[idx]
            # Randomly select a new pitch within MIDI note range (21 to 108 for piano)
            new_pitch = np.random.randint(21, 109)
            instrument.notes[idx].pitch = new_pitch
    return pm

def process_midi_files(data_dir, output_dir, noise_levels):
    """Process all MIDI files in the dataset by randomizing note pitches."""
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
            
            # Load MIDI file
            pm = pretty_midi.PrettyMIDI(midi_file)
            
            # Set instrument to Acoustic Grand Piano (Program Number 0)
            for instrument in pm.instruments:
                instrument.program = 0
                instrument.is_drum = False  # Ensure it's not a drum instrument
            
            # Create versions with different noise levels
            for noise_percent in noise_levels:
                noise_level = noise_percent / 100.0  # Convert to 0.0 - 1.0
                
                if noise_level == 0:
                    # Use original MIDI data
                    pm_noisy = pm
                else:
                    # Create a deep copy to avoid modifying the original
                    pm_noisy = copy.deepcopy(pm)
                    # Randomize note pitches
                    pm_noisy = randomize_midi_notes(pm_noisy, noise_level)
                
                # Convert MIDI to audio
                y_noisy, sr = midi_to_audio(pm_noisy)
                
                # Normalize the audio
                max_val = np.max(np.abs(y_noisy))
                if max_val > 0:
                    y_noisy = y_noisy / max_val
                
                # Save the audio file
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
    output_directory = 'EMOPIA_1.0_noisy_midi_randomization'
    noise_levels = [0, 25, 50, 75, 100]  # Adjust as needed
    
    process_midi_files(data_directory, output_directory, noise_levels)
