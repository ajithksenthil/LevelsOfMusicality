import os
import numpy as np
import copy
import pretty_midi
import soundfile as sf

def get_pitch_range(instrument, padding=12):
    """
    Get the pitch range of notes in an instrument track.
    Returns min and max pitch with optional padding (in semitones).
    """
    if not instrument.notes:
        return 21, 108  # Default piano range if no notes
    
    pitches = [note.pitch for note in instrument.notes]
    min_pitch = max(21, min(pitches) - padding)  # Don't go below piano range
    max_pitch = min(108, max(pitches) + padding)  # Don't go above piano range
    return min_pitch, max_pitch

def randomize_midi(pm, noise_level, preserve_range=True):
    """
    Randomize pitches, start times, and durations of notes in the MIDI data based on noise_level.
    noise_level ranges from 0.0 (no changes) to 1.0 (all notes randomized).
    preserve_range: If True, constrains new pitches to be within the original pitch range (±1 octave).
    """
    for instrument in pm.instruments:
        num_notes = len(instrument.notes)
        if num_notes == 0:
            continue

        # Get the original pitch range if preserving range
        if preserve_range:
            min_pitch, max_pitch = get_pitch_range(instrument, padding=12)  # ±1 octave padding
        else:
            min_pitch, max_pitch = 21, 108  # Full piano range

        # Calculate the number of notes to randomize
        num_notes_to_randomize = int(noise_level * num_notes)

        # Randomly select notes to randomize for each attribute
        indices_pitch = np.random.choice(num_notes, num_notes_to_randomize, replace=False)
        indices_timing = np.random.choice(num_notes, num_notes_to_randomize, replace=False)
        indices_duration = np.random.choice(num_notes, num_notes_to_randomize, replace=False)

        # Store original pitches for reference
        original_pitches = [note.pitch for note in instrument.notes]
        pitch_mean = np.mean(original_pitches)
        pitch_std = np.std(original_pitches)

        # Randomize pitches while preserving general range
        for idx in indices_pitch:
            original_note = instrument.notes[idx]
            if preserve_range:
                # Use a normal distribution centered around the original pitch
                new_pitch = int(round(np.random.normal(
                    loc=original_note.pitch,
                    scale=pitch_std * noise_level
                )))
                # Ensure the new pitch stays within the allowed range
                new_pitch = max(min_pitch, min(max_pitch, new_pitch))
            else:
                # Original behavior (completely random within piano range)
                new_pitch = np.random.randint(min_pitch, max_pitch + 1)
            
            instrument.notes[idx].pitch = new_pitch

        # Randomize start times (keeping original logic)
        max_shift = 0.5 * noise_level  # Scale the maximum shift by noise level
        for idx in indices_timing:
            original_note = instrument.notes[idx]
            shift = np.random.uniform(-max_shift, max_shift)
            new_start = original_note.start + shift
            new_start = max(0, new_start)
            duration = original_note.end - original_note.start
            new_end = new_start + duration
            instrument.notes[idx].start = new_start
            instrument.notes[idx].end = new_end

        # Randomize durations (keeping original logic)
        max_change = 0.5 * noise_level  # Scale the maximum duration change by noise level
        for idx in indices_duration:
            original_note = instrument.notes[idx]
            change = np.random.uniform(-max_change, max_change)
            new_duration = (original_note.end - original_note.start) + change
            new_duration = max(0.1, new_duration)
            instrument.notes[idx].end = original_note.start + new_duration

    return pm

def midi_to_audio(pm, fs=44100):
    """Convert PrettyMIDI object to audio using the built-in synthesizer."""
    audio = pm.synthesize(fs=fs)
    return audio, fs

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
            
            # Set instrument to Acoustic Grand Piano
            for instrument in pm_original.instruments:
                instrument.program = 0
                instrument.is_drum = False
            
            # Create versions with different noise levels
            for noise_percent in noise_levels:
                noise_level = noise_percent / 100.0
                
                if noise_level == 0:
                    pm_noisy = pm_original
                else:
                    pm_noisy = copy.deepcopy(pm_original)
                    pm_noisy = randomize_midi(pm_noisy, noise_level, preserve_range=True)
                
                # Convert MIDI to audio
                y_noisy, sr = midi_to_audio(pm_noisy)
                
                # Normalize the audio
                max_val = np.max(np.abs(y_noisy))
                if max_val > 0:
                    y_noisy = y_noisy / max_val
                
                # Clip the audio
                total_samples = len(y_noisy)
                clip_samples = int(clip_duration * sr)
                
                if clip_samples < total_samples:
                    start_idx = (total_samples - clip_samples) // 2
                    end_idx = start_idx + clip_samples
                    y_noisy = y_noisy[start_idx:end_idx]
                else:
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
    output_directory = 'cons_5s_EMOPIA_1.0_noisy_midi_randomization_comb_experimental_stimuli'
    noise_levels = [0, 25, 50, 75, 100]
    clip_duration = 5.0

    process_midi_files(data_directory, output_directory, noise_levels, clip_duration)

"""
New notes will stay within approximately ±1 octave of the original pitch range
Uses a normal distribution instead of uniform random, making extreme jumps less likely
Preserves the general pitch characteristics of the original piece
Scales all randomization (pitch, timing, duration) with the noise level
"""