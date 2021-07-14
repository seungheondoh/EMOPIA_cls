import os
import glob
import copy
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import miditoolkit
from miditoolkit.midi import parser as mid_parser
from miditoolkit.pianoroll import parser as pr_parser
from miditoolkit.midi.containers import Marker, Instrument, TempoChange
import collections
from chorder import Dechorder
from datetime import datetime


# ================================================== #  
#  Configuration                                     #
# ================================================== #  
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4
INSTR_NAME_MAP = {'piano': 0}
MIN_BPM = 40
MIN_VELOCITY = 40
NOTE_SORTING = 1 #  0: ascending / 1: descending

DEFAULT_VELOCITY_BINS = np.linspace(0,  128, 64+1, dtype=np.int)
DEFAULT_BPM_BINS      = np.linspace(32, 224, 64+1, dtype=np.int)
DEFAULT_SHIFT_BINS    = np.linspace(-60, 60, 60+1, dtype=np.int)
DEFAULT_DURATION_BINS = np.arange(
        BEAT_RESOL/8, BEAT_RESOL*8+1, BEAT_RESOL/8)

num2pitch = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#',
    9: 'A',
    10: 'A#',
    11: 'B',
}


def traverse_dir(
        root_dir,
        extension=('mid', 'MID', 'midi'),
        amount=None,
        str_=None,
        is_pure=False,
        verbose=False,
        is_sort=False,
        is_ext=True):
    if verbose:
        print('[*] Scanning...')
    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                if (amount is not None) and (cnt == amount):
                    break
                if str_ is not None:
                    if str_ not in file:
                        continue
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir):] if is_pure else mix_path
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                if verbose:
                    print(pure_path)
                file_list.append(pure_path)
                cnt += 1
    if verbose:
        print('Total: %d files' % len(file_list))
        print('Done!!!')
    if is_sort:
        file_list.sort()
    return file_list

def analyzer(path_infile):
    #print('----')
    #print(' >', path_infile)
    #print(' >', path_outfile)

    # load
    midi_obj = miditoolkit.midi.parser.MidiFile(path_infile)
    midi_obj_out = copy.deepcopy(midi_obj)
    notes = midi_obj.instruments[0].notes
    notes = sorted(notes, key=lambda x: (x.start, x.pitch))

    # --- chord --- #
    # exctract chord
    chords = Dechorder.dechord(midi_obj)
    markers = []
    for cidx, chord in enumerate(chords):
        if chord.is_complete():
            chord_text = num2pitch[chord.root_pc] + '_' + chord.quality + '_' + num2pitch[chord.bass_pc]
        else:
            chord_text = 'N_N_N'
        markers.append(Marker(time=int(cidx*480), text=chord_text))

    # dedup
    prev_chord = None
    dedup_chords = []
    for m in markers:
        if m.text != prev_chord:
            prev_chord = m.text
            dedup_chords.append(m)

    # --- global properties --- #
    # global tempo
    tempos = [b.tempo for b in midi_obj.tempo_changes][:40]
    tempo_median = np.median(tempos)
    global_bpm =int(tempo_median)
    #print(' > [global] bpm:', global_bpm)

    # markers
    midi_obj_out.markers = dedup_chords
    midi_obj_out.markers.insert(0, Marker(text='global_bpm_'+str(int(global_bpm)), time=0))

    # save
    midi_obj_out.instruments[0].name = 'piano'
    return midi_obj_out

def corpus(midi_obj):
    # load notes
    instr_notes = collections.defaultdict(list)
    for instr in midi_obj.instruments:
        # skip 
        if instr.name not in INSTR_NAME_MAP.keys():
            continue

        # process
        instr_idx = INSTR_NAME_MAP[instr.name]
        for note in instr.notes:
            note.instr_idx=instr_idx
            instr_notes[instr_idx].append(note)
        if NOTE_SORTING == 0:
            instr_notes[instr_idx].sort(
                key=lambda x: (x.start, x.pitch))
        elif NOTE_SORTING == 1:
            instr_notes[instr_idx].sort(
                key=lambda x: (x.start, -x.pitch))
        else:
            raise ValueError(' [x] Unknown type of sorting.')

    # load chords
    chords = []
    for marker in midi_obj.markers:
        if marker.text.split('_')[0] != 'global' and \
        'Boundary' not in marker.text.split('_')[0]:
            chords.append(marker)
    chords.sort(key=lambda x: x.time)

    # load tempos
    tempos = midi_obj.tempo_changes
    tempos.sort(key=lambda x: x.time)

    # load labels
    labels = []
    for marker in midi_obj.markers:
        if 'Boundary' in marker.text.split('_')[0]:
            labels.append(marker)
    labels.sort(key=lambda x: x.time)

    # load global bpm
    gobal_bpm = 120
    for marker in midi_obj.markers:
        if marker.text.split('_')[0] == 'global' and \
            marker.text.split('_')[1] == 'bpm':
            gobal_bpm = int(marker.text.split('_')[2])
        
    # --- process items to grid --- #
    # compute empty bar offset at head
    first_note_time = min([instr_notes[k][0].start for k in instr_notes.keys()])
    last_note_time = max([instr_notes[k][-1].start for k in instr_notes.keys()])

    quant_time_first = int(np.round(first_note_time  / TICK_RESOL) * TICK_RESOL)
    offset = quant_time_first // BAR_RESOL # empty bar
    last_bar = int(np.ceil(last_note_time / BAR_RESOL)) - offset
    #print(' > offset:', offset)
    #print(' > last_bar:', last_bar)

    # process notes
    intsr_gird = dict()
    for key in instr_notes.keys():
        notes = instr_notes[key]
        note_grid = collections.defaultdict(list)
        for note in notes:
            note.start = note.start - offset * BAR_RESOL
            note.end = note.end - offset * BAR_RESOL

            # quantize start
            quant_time = int(np.round(note.start / TICK_RESOL) * TICK_RESOL)

            # velocity
            note.velocity = DEFAULT_VELOCITY_BINS[
                np.argmin(abs(DEFAULT_VELOCITY_BINS-note.velocity))]
            note.velocity = max(MIN_VELOCITY, note.velocity)

            # shift of start
            note.shift = note.start - quant_time 
            note.shift = DEFAULT_SHIFT_BINS[np.argmin(abs(DEFAULT_SHIFT_BINS-note.shift))]

            # duration
            note_duration = note.end - note.start
            if note_duration > BAR_RESOL:
                note_duration = BAR_RESOL
            ntick_duration = int(np.round(note_duration / TICK_RESOL) * TICK_RESOL)
            note.duration = ntick_duration

            # append
            note_grid[quant_time].append(note)
        
        # set to track
        intsr_gird[key] = note_grid.copy()

    # process chords
    chord_grid = collections.defaultdict(list)
    for chord in chords:
        # quantize
        chord.time = chord.time - offset * BAR_RESOL
        chord.time  = 0 if chord.time < 0 else chord.time 
        quant_time = int(np.round(chord.time / TICK_RESOL) * TICK_RESOL)

        # append
        chord_grid[quant_time].append(chord)

    # process tempo
    tempo_grid = collections.defaultdict(list)
    for tempo in tempos:
        # quantize
        tempo.time = tempo.time - offset * BAR_RESOL
        tempo.time = 0 if tempo.time < 0 else tempo.time
        quant_time = int(np.round(tempo.time / TICK_RESOL) * TICK_RESOL)
        tempo.tempo = DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS-tempo.tempo))]

        # append
        tempo_grid[quant_time].append(tempo)

    # process boundary
    label_grid = collections.defaultdict(list)
    for label in labels:
        # quantize
        label.time = label.time - offset * BAR_RESOL
        label.time = 0 if label.time < 0 else label.time
        quant_time = int(np.round(label.time / TICK_RESOL) * TICK_RESOL)

        # append
        label_grid[quant_time] = [label]
        
    # process global bpm
    gobal_bpm = DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS-gobal_bpm))]

    # collect
    song_data = {
        'notes': intsr_gird,
        'chords': chord_grid,
        'tempos': tempo_grid,
        'labels': label_grid,
        'metadata': {
            'global_bpm': gobal_bpm,
            'last_bar': last_bar,
        }
    }
    
    return song_data

# define event
def create_event(name, value):
    event = dict()
    event['name'] = name
    event['value'] = value
    return event

# load key from csv
def get_key(key_data, fn):
    key_map = {}
    df = pd.read_csv(key_data)
    filename = list(df['name'])
    key = list(df['keyname'])
    for i, filename in enumerate(filename):
        key_map[filename] = key[i]
        
    return key_map[fn]

# core functions
def event(data, key = None):
    '''
    <<< REMI v2 >>>
    task: 2 track 
        1: piano      (note + tempo + chord)
    ---
    remove duplicate position tokens
    '''    
    # global tag
    global_end = data['metadata']['last_bar'] * BAR_RESOL

    # process
    final_sequence = []
    for bar_step in range(0, global_end, BAR_RESOL):
        final_sequence.append(create_event('Bar', None))
        if key == None:
            pass
        else:
            final_sequence.append(create_event('Key', key))

        # --- piano track --- #
        for timing in range(bar_step, bar_step + BAR_RESOL, TICK_RESOL):
            pos_events = []

            # unpack
            t_chords = data['chords'][timing]
            t_tempos = data['tempos'][timing]
            t_notes = data['notes'][0][timing] # piano track

            # chord
            if len(t_chords):
                root, quality, bass = t_chords[0].text.split('_')
                pos_events.append(create_event('Chord', root+'_'+quality))

            # tempo
            if len(t_tempos):
                pos_events.append(create_event('Tempo', t_tempos[0].tempo))

            # note 
            if len(t_notes):
                for note in t_notes:
                    pos_events.extend([
                        create_event('Note_Pitch', note.pitch),
                        create_event('Note_Velocity', note.velocity),
                        create_event('Note_Duration', note.duration),
                    ])

            # collect & beat
            if len(pos_events):
                final_sequence.append(
                    create_event('Beat', (timing-bar_step)//TICK_RESOL))
                final_sequence.extend(pos_events)

    # BAR ending
    final_sequence.append(create_event('Bar', None))   

    # EOS
    final_sequence.append(create_event('EOS', None))   

    return final_sequence


def main():
    midi_path = '/home/joann8512/NAS_189/home/PEmoDataset/midis/Q1__8v0MFBZoco_0.mid'
    key_data = '../src/key_mode_tempo.csv'
    path_outdir = '../test/events'
    os.makedirs(path_outdir, exist_ok=True)
    
    # get key from key_data
    fn = midi_path.split('/')[-1]
    key = get_key(key_data, os.path.splitext(fn)[0])
    
    midi_obj = analyzer(midi_path)
    song_data = corpus(midi_obj)
    final_sequence = event(song_data, key)
    
    # save
    pickle.dump(final_sequence, open(os.path.join(path_outdir, fn+'.pkl'), 'wb'))
    
    
if __name__ == '__main__':
    main()