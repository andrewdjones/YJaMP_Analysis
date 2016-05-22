The Yale Jazz Midi Piano corpus (YJaMP) consists of 189 solo jazz tracks played on a baby grand piano with MIDI output.  For access to the data, please contact andrew.d.jones@yale.edu.

This repository holds code written for the YJaMP corpus and documented in my forthcoming dissertation at Yale, "The Statistical Temporality of Harmony: Jazz Syntax from Corpus Analytics."

--------------

The most dissertation-relevant code with the best maintenance is found in the following files, which roughly parallel the chapter structure of the dissertation:

jazzKey.py: code for extracting/testing key profiles and producing locally-transposed voicings and scale degree sets from the raw MIDI data.  Turns raw data into chord objects of various kinds.

objects_syntax.py: code for locating voicings and scale degree sets in terms of one another and the means to extract forwards and backwards-directed temporal syntax from the corpus.  Compiles statistics regarding chord objects and their temporal behavior.

yjampClus.py: meatier code for k-medoids, agglomerative, flat hierarchical, and sub/meta clustering of harmonic data, with and without dimensional reduction by PCA.  Clusters chord objects to extract syntax based on similar temporal behavior at multiple time scales.

catTPD.py: PCA-based agglomerative clustering and dendrogram models.  Bootstraps categories of harmonic progression, building them from transition similarity and without any required knowledge of pitch similarity.

--------------

Other .py files here provide more exploratory (and less curated) code for YJaMP (and should be approached with caution).

chordSmusher.py: contains code to make chords out of messy midi data in alternate ways.

PCsetCodes.py and VoicingCodes.py: the sandboxes from which objects_syntax.py came; contains scripts for extracting and manipulating pitch class sets and (midi-based, non-octave-reduced) voicings, respectively.

jazzSuperset.py: within the framework of music21, allows reduction of raw voicing types to superset relations.  May capture certain jazz hand position norms that violate strict simultaneity and closeness in pitch space.

miditools.py: contains deprecated versions of code now found in jazzKey.py and inter-onset codes for use on raw MIDI files.