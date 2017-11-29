##!/usr/bin/python
#
import midi
#
#
## Instantiate a MIDI Pattern (contains a list of tracks)
#pattern = midi.Pattern()
## Instantiate a MIDI Track (contains a list of MIDI events)
#track = midi.Track()
## Append the track to the pattern
#pattern.append(track)
#
#
#on = midi.NoteOnEvent(tick=0, velocity=95, pitch=notes)
##on2 = midi.NoteOnEvent(tick=0, velocity=95, pitch=midi.E_3)
##on3 = midi.NoteOnEvent(tick=0, velocity=95, pitch=midi.G_3)
#
#off = midi.NoteOffEvent(tick=100, pitch=notes)
##off2 = midi.NoteOffEvent(tick=0, pitch=midi.E_3)
##off3 = midi.NoteOffEvent(tick=0, pitch=midi.G_3)
#
#track.append(on)
##track.append(on2)
##track.append(on3)
#
#track.append(off)
##track.append(off2)
##track.append(off3)
#
##on = midi.NoteOnEvent(tick=0, velocity=95, pitch=midi.C_3)
##on2 = midi.NoteOnEvent(tick=0, velocity=95, pitch=midi.E_3)
##on3 = midi.NoteOnEvent(tick=0, velocity=95, pitch=midi.G_3)
##
##off = midi.NoteOffEvent(tick=100, pitch=midi.C_3)
##off2 = midi.NoteOffEvent(tick=0, pitch=midi.E_3)
##off3 = midi.NoteOffEvent(tick=0, pitch=midi.G_3)
##
##track.append(on)
##track.append(on2)
##track.append(on3)
##
##track.append(off)
##track.append(off2)
##track.append(off3)
#
#
#
#
#
##on = midi.NoteOnEvent(tick=100, velocity=95, pitch=midi.G_3)
##on2 = midi.NoteOnEvent(tick=100, velocity=95, pitch=midi.E_3)
##track.append(on)
##track.append(on2)
##off = midi.NoteOffEvent(tick=200, pitch=midi.G_3)
##off2 = midi.NoteOffEvent(tick=200, pitch=midi.E_3)
##track.append(off)
##track.append(off2)
#
## Add the end of track event, append it to the track
#eot = midi.EndOfTrackEvent(tick=1)
#track.append(eot)
## Print out the pattern
#print pattern
## Save the pattern to disk
#midi.write_midifile("example.mid", pattern)



####################################################################################################################################





def midi_var(value):
	
	if value == 'A1':
		note = midi.A_1
	elif value == 'A2':
		note = midi.A_2
	elif value == 'A3':
		note = midi.A_3
	elif value == 'A4':
		note = midi.A_4
	elif value == 'A5':
		note = midi.A_5
	elif value == 'A6':
		note = midi.A_6
	elif value == 'A7':
		note = midi.A_7
	elif value == 'A8':
		note = midi.A_8
		
	elif value == 'A#1':
		note = midi.A_1
	elif value == 'A#2':
		note = midi.A_2
	elif value == 'A#3':
		note = midi.A_3
	elif value == 'A#4':
		note = midi.A_4
	elif value == 'A#5':
		note = midi.A_5
	elif value == 'A#6':
		note = midi.A_6
	elif value == 'A#7':
		note = midi.A_7
	elif value == 'A#8':
		note = midi.A_8
		
	elif value == 'B1':
		note = midi.B_1
	elif value == 'B2':
		note = midi.B_2
	elif value == 'B3':
		note = midi.B_3
	elif value == 'B4':
		note = midi.B_4
	elif value == 'B5':
		note = midi.B_5
	elif value == 'B6':
		note = midi.B_6
	elif value == 'B7':
		note = midi.B_7
	elif value == 'B8':
		note = midi.B_8
		
	elif value == 'C1':
		note = midi.C_1	
	elif value == 'C2':
		note = midi.C_2
	elif value == 'C3':
		note = midi.C_3
	elif value == 'C4':
		note = midi.C_4
	elif value == 'C5':
		note = midi.C_5
	elif value == 'C6':
		note = midi.C_6
	elif value == 'C7':
		note = midi.C_7
	elif value == 'C8':
		note = midi.C_8

	elif value == 'C#1':
		note = midi.Cs_1	
	elif value == 'C#2':
		note = midi.Cs_2
	elif value == 'C#3':
		note = midi.Cs_3
	elif value == 'C#4':
		note = midi.Cs_4
	elif value == 'C#5':
		note = midi.Cs_5
	elif value == 'C#6':
		note = midi.Cs_6
	elif value == 'C#7':
		note = midi.Cs_7
	elif value == 'C#8':
		note = midi.Cs_8

	elif value == 'D1':
		note = midi.D_1
	elif value == 'D2':
		note = midi.D_2
	elif value == 'D3':
		note = midi.D_3
	elif value == 'D4':
		note = midi.D_4
	elif value == 'D5':
		note = midi.D_5
	elif value == 'D6':
		note = midi.D_6
	elif value == 'D7':
		note = midi.D_7
	elif value == 'D8':
		note = midi.D_8

	elif value == 'D#1':
		note = midi.Ds_1
	elif value == 'D#2':
		note = midi.Ds_2
	elif value == 'D#3':
		note = midi.Ds_3
	elif value == 'D#4':
		note = midi.Ds_4
	elif value == 'D#5':
		note = midi.Ds_5
	elif value == 'D#6':
		note = midi.Ds_6
	elif value == 'D#7':
		note = midi.Ds_7
	elif value == 'D#8':
		note = midi.Ds_8

		
	elif value == 'E1':
		note = midi.E_1
	elif value == 'E2':
		note = midi.E_2
	elif value == 'E3':
		note = midi.E_3
	elif value == 'E4':
		note = midi.E_4
	elif value == 'E5':
		note = midi.E_5
	elif value == 'E6':
		note = midi.E_6
	elif value == 'E7':
		note = midi.E_7
	elif value == 'E8':
		note = midi.E_8
		
	elif value == 'F1':
		note = midi.F_1
	elif value == 'F2':
		note = midi.F_2
	elif value == 'F3':
		note = midi.F_3
	elif value == 'F4':
		note = midi.F_4
	elif value == 'F5':
		note = midi.F_5
	elif value == 'F6':
		note = midi.F_6
	elif value == 'F7':
		note = midi.F_7
	elif value == 'F8':
		note = midi.F_8

	elif value == 'F#1':
		note = midi.Fs_1
	elif value == 'F#2':
		note = midi.Fs_2
	elif value == 'F#3':
		note = midi.Fs_3
	elif value == 'F#4':
		note = midi.Fs_4
	elif value == 'F#5':
		note = midi.Fs_5
	elif value == 'F#6':
		note = midi.Fs_6
	elif value == 'F#7':
		note = midi.Fs_7
	elif value == 'F#8':
		note = midi.Fs_8

		
	elif value == 'G1':
		note = midi.G_1
	elif value == 'G2':
		note = midi.G_2
	elif value == 'G3':
		note = midi.G_3
	elif value == 'G4':
		note = midi.G_4
	elif value == 'G5':
		note = midi.G_5
	elif value == 'G6':
		note = midi.G_6
	elif value == 'G7':
		note = midi.G_7
	elif value == 'G8':
		note = midi.G_8

	elif value == 'G#1':
		note = midi.Gs_1
	elif value == 'G#2':
		note = midi.Gs_2
	elif value == 'G#3':
		note = midi.Gs_3
	elif value == 'G#4':
		note = midi.Gs_4
	elif value == 'G#5':
		note = midi.Gs_5
	elif value == 'G#6':
		note = midi.Gs_6
	elif value == 'G#7':
		note = midi.Gs_7
	elif value == 'G#8':
		note = midi.Gs_8
		
	return note







def make_midi(song_array):
	
	pattern = midi.Pattern()
	track = midi.Track()
	pattern.append(track)
	for chord in song_array:
		offarray = []
		chordcount = 0
		count = 0

		for note in chord:
			print note
			if count == 0:
				on = midi.NoteOnEvent(tick=chordcount*100, velocity=95, pitch=midi_var(note))
				off = midi.NoteOffEvent(tick=((chordcount*100)+100), pitch=midi_var(note))
				track.append(on)
				offarray.append(off)
				count += 1
				
			else:
				on = midi.NoteOnEvent(tick=chordcount*100, velocity=95, pitch=midi_var(note))
				off = midi.NoteOffEvent(tick=chordcount*100, pitch=midi_var(note))
				track.append(on)
				offarray.append(off)
				count+=1
				
		chordcount += 1

		for off in offarray:
			track.append(off)
				
	# Add the end of track event, append it to the track
	eot = midi.EndOfTrackEvent(tick=1)
	track.append(eot)
	# Print out the pattern
	print track
#	print pattern
	# Save the pattern to disk
	midi.write_midifile("example.mid", pattern)

		
make_midi([['A3','C#3','E3'],['C3','E3','G3','B3'],['B3','D#3','F#3']])
				
		

def add_note(note):
	on = midi.NoteOnEvent(tick=0, velocity=95, pitch=midi.G_3)
	track.append(on)
	off = midi.NoteOffEvent(tick=100, pitch=midi.G_3)
	track.append(off)








