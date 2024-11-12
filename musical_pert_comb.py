import os
import numpy as np
import copy  # Import the copy module
import pretty_midi
import soundfile as sf

def midi_to_audio(pm, fs=44100):
    """Convert PrettyMIDI object to audio using the built-in synthesizer."""
    audio = pm.synthesize(fs=fs)
    return audio, fs

def randomize_midi(pm, noise_level):
    """
    Randomize pitches, start times, and durations of notes in the MIDI data based on noise_level.
    noise_level ranges from 0.0 (no changes) to 1.0 (all notes randomized).
    """
    for instrument in pm.instruments:
        num_notes = len(instrument.notes)
        if num_notes == 0:
            continue  # Skip if there are no notes

        # Calculate the number of notes to randomize
        num_notes_to_randomize = int(noise_level * num_notes)

        # Randomly select notes to randomize for each attribute
        indices_pitch = np.random.choice(num_notes, num_notes_to_randomize, replace=False)
        indices_timing = np.random.choice(num_notes, num_notes_to_randomize, replace=False)
        indices_duration = np.random.choice(num_notes, num_notes_to_randomize, replace=False)

        # Randomize pitches
        for idx in indices_pitch:
            original_note = instrument.notes[idx]
            # Randomly select a new pitch within MIDI note range (21 to 108 for piano)
            new_pitch = np.random.randint(21, 109)
            instrument.notes[idx].pitch = new_pitch

        # Randomize start times
        max_shift = 0.5  # Maximum shift in seconds (adjust as needed)
        for idx in indices_timing:
            original_note = instrument.notes[idx]
            shift = np.random.uniform(-max_shift, max_shift)
            new_start = original_note.start + shift
            # Ensure the new start time is non-negative
            new_start = max(0, new_start)
            # Update end time to maintain duration
            duration = original_note.end - original_note.start
            new_end = new_start + duration
            instrument.notes[idx].start = new_start
            instrument.notes[idx].end = new_end

        # Randomize durations
        max_change = 0.5  # Maximum change in seconds (adjust as needed)
        for idx in indices_duration:
            original_note = instrument.notes[idx]
            change = np.random.uniform(-max_change, max_change)
            new_duration = (original_note.end - original_note.start) + change
            # Ensure the new duration is positive
            new_duration = max(0.1, new_duration)
            instrument.notes[idx].end = original_note.start + new_duration

    return pm

def process_midi_files(data_dir, output_dir, noise_levels, clip_duration):
    """Process all MIDI files in the dataset by randomizing note attributes and clipping audio."""
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
            pm_original = pretty_midi.PrettyMIDI(midi_file)
            
            # Set instrument to Acoustic Grand Piano (Program Number 0)
            for instrument in pm_original.instruments:
                instrument.program = 0
                instrument.is_drum = False  # Ensure it's not a drum instrument
            
            # Create versions with different noise levels
            for noise_percent in noise_levels:
                noise_level = noise_percent / 100.0  # Convert to 0.0 - 1.0
                
                if noise_level == 0:
                    # Use original MIDI data
                    pm_noisy = pm_original
                else:
                    # Create a deep copy to avoid modifying the original
                    pm_noisy = copy.deepcopy(pm_original)
                    # Randomize note attributes
                    pm_noisy = randomize_midi(pm_noisy, noise_level)
                
                # Convert MIDI to audio
                y_noisy, sr = midi_to_audio(pm_noisy)
                
                # Normalize the audio
                max_val = np.max(np.abs(y_noisy))
                if max_val > 0:
                    y_noisy = y_noisy / max_val
                
                # Clip the audio to the specified duration
                total_samples = len(y_noisy)
                clip_samples = int(clip_duration * sr)
                
                if clip_samples < total_samples:
                    # Calculate start and end indices for clipping
                    start_idx = (total_samples - clip_samples) // 2
                    end_idx = start_idx + clip_samples
                    y_noisy = y_noisy[start_idx:end_idx]
                else:
                    # If the audio is shorter than the clip duration, pad it with zeros
                    padding = clip_samples - total_samples
                    pad_left = padding // 2
                    pad_right = padding - pad_left
                    y_noisy = np.pad(y_noisy, (pad_left, pad_right), mode='constant')
                
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
    data_directory = 'InputData'
    output_directory = '5s_EMOPIA_1.0_noisy_midi_randomization_comb_experimental_stimuli'
    noise_levels = [0, 25, 50, 75, 100]  # Adjust as needed
    clip_duration = 5.0  # Desired duration in seconds

    process_midi_files(data_directory, output_directory, noise_levels, clip_duration)
