chordSmusher.py is the smallest of the files, and it contains code to make chords out of messy midi data.

The rest vary.  jazzKey.py, for example, does a bunch of sequential stuff broken up.

Codeflow for jazzKey.py:

For locally-transposed scale-degree sets:
1. In jazzKey.py, run chordFinder(wsize).  This gets you scale degree sets (transposed to local keys) and keys (indexed by millisecond time stamps).
2. In jazzKey.py, run tallySDSets(cmin).  This gets you a breakdown of the most common scale degree sets.
3. If you want to find out what comes n(<numWin) windows after a given SDSet, run nAfterSDS(sds, numWin).

Notes:
The above codes will invoke midiTimeWindows(), jazzKeyFinder(), and perplexity().
I'm not totally sure what transPCVecs() does anymore, but I think it's a deprecated version of chordFinder().
The things above midiTimeWindows() in the .py file are mostly deprecated and have to do with music21 versions of the same operations.