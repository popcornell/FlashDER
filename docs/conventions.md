# Differences between scoring tools

NIST `md-eval-22.pl`, dscore and pyannote.metrics all compute the diarization error rate (DER), and dscore and pyannote.metrics also compute the Jaccard error rate (JER). They make different choices in several places, and the same input can receive different scores. This document lists each difference with a small example and the value reported by each tool.

The values were obtained with `md-eval-22.pl` (version 22, as distributed with dscore), dscore at commit `e02f949` and pyannote.metrics 4.0.0. Times are in seconds and scores in percent. Every example is reproduced in `tests/test_conventions.py`, which checks the flashder values and, when the tools are installed, the values of the original tools.

flashder reproduces each tool through the `style` argument (`md-eval`, `dscore` or `pyannote`). The default is `md-eval`. md-eval does not compute JER, so in the `md-eval` style flashder computes JER with the dscore definition, but without frames.

In the examples, `ref` and `hyp` list turns as `speaker [start, end]`.

## 1. Collar width

md-eval (`-c`) and dscore (`--collar`) exclude a region of the given length on each side of every reference boundary. pyannote.metrics (`collar=`) takes the total width of that region, half on each side.

```
ref  A [0, 10]
hyp  x [0.2, 10]
uem  [0, 10]
```

| tool and setting | DER |
|---|---|
| md-eval `-c 0.25` | 0.00 |
| dscore `--collar 0.25` | 0.00 |
| pyannote `collar=0.5` | 0.00 |
| pyannote `collar=0.25` | 0.77 |

With pyannote `collar=0.25` the excluded region around time 0 is [-0.125, 0.125], and [0.125, 0.2] is counted as missed speech. In flashder the collar is always given per side, as in md-eval: `collar=0.25` corresponds to pyannote `collar=0.5`.

## 2. Default UEM

When no UEM is given, md-eval scores the extent of the reference. dscore and pyannote.metrics score the extent of the reference and hypothesis together.

```
ref  A [0, 10]
hyp  x [0, 12]
no UEM
```

| tool | DER | JER |
|---|---|---|
| md-eval | 0.00 | |
| dscore | 20.00 | 16.67 |
| pyannote.metrics | 20.00 | 16.67 |

The hypothesis speech in [10, 12] is outside the region scored by md-eval.

## 3. Region used for the speaker mapping

md-eval and dscore compute the optimal mapping between reference and hypothesis speakers on the whole UEM, before collars and overlap regions are removed. pyannote.metrics computes it on the scored region only. The two can produce different mappings when collars or overlap exclusion remove most of a speaker's speech.

```
ref  A [0, 1] [2, 3] [4, 5] [6, 7]
     B [8, 10]
hyp  x [0, 1] [2, 3] [4, 5] [6, 7] [8, 10]
uem  [0, 10]
collar 0.5 s per side
```

| tool | DER | JER |
|---|---|---|
| md-eval `-c 0.5` | 100.00 | |
| dscore `--collar 0.5` | 100.00 | 66.67 |
| pyannote `collar=1.0` | 0.00 | 0.00 |

On the whole UEM, x co-occurs with A for 4 s and with B for 2 s, so md-eval maps x to A. The collars remove all of A's turns and leave [8.5, 9.5], where only B speaks. md-eval counts this second as confusion. pyannote.metrics maps x to B, the only speaker in the scored region, and reports no error.

## 4. Turns cut by the UEM

dscore crops all turns to the UEM before calling md-eval. A turn that extends past the end of a UEM region therefore ends at the UEM boundary, and md-eval places a collar there. md-eval and pyannote.metrics, given the uncropped turns, place no collar at that point.

```
ref  A [0, 10]
hyp  x [0, 4.8]
uem  [0, 5]
collar 0.25 s per side
```

| tool | DER |
|---|---|
| md-eval `-c 0.25` | 4.21 |
| dscore `--collar 0.25` | 0.00 |
| pyannote `collar=0.5` | 4.21 |

md-eval scores [0.25, 5] and counts [4.8, 5] as missed (0.2 / 4.75). dscore also excludes [4.75, 5] and reports no error.

## 5. Overlapping turns of the same speaker

Reference files sometimes contain two turns of the same speaker that overlap in time. The tools treat them in three different ways.

- md-eval counts the speaker once when computing errors, but counts both turns when it decides where reference overlap occurs (`-1`).
- dscore merges the turns before scoring.
- pyannote.metrics counts the speaker once per turn, both for errors and for overlap.

```
ref  A [0, 2]
     A [1, 3]
hyp  x [0, 2.5]
uem  [0, 3]
```

| tool | DER | DER, overlap excluded |
|---|---|---|
| md-eval | 16.67 | 25.00 |
| dscore | 16.67 | 16.67 |
| pyannote.metrics | 37.50 | 25.00 |

Without overlap exclusion, md-eval and dscore score 3 s of reference speech with 0.5 s missed. pyannote.metrics counts A twice in [1, 2], which gives 4 s of reference speech and 1.5 s missed. With overlap exclusion, md-eval and pyannote.metrics treat [1, 2] as overlap and remove it (0.5 / 2). dscore has merged the turns, finds no overlap and scores all 3 s (0.5 / 3).

flashder counts a speaker once in every style. `count_overlapping_turns=True` reproduces the pyannote.metrics value. The overlap region is determined from turns in the `md-eval` and `pyannote` styles and from merged turns in the `dscore` style, as in the original tools.

Hypothesis turns are subject to the same difference: pyannote.metrics counts overlapping turns of one hypothesis speaker more than once. This is the cause of the difference in DER between pyannote.metrics and the other tools in the benchmarks of the README.

## 6. Recordings without reference speech

md-eval only scores recordings that appear in the reference. dscore reports a DER of 100 percent for such a recording when the hypothesis contains speech, but its overall DER is taken from md-eval and excludes it. pyannote.metrics scores every recording passed to it.

```
recording 1   ref A [0, 10]   hyp x [0, 10]   uem [0, 10]
recording 2   no reference    hyp y [0, 5]    uem [0, 5]
```

| tool | overall DER |
|---|---|
| md-eval | 0.00 |
| dscore | 0.00 |
| pyannote.metrics | 50.00 |

pyannote.metrics adds 5 s of false alarm to 10 s of reference speech. Its `JaccardErrorRate` raises `ZeroDivisionError` on recording 2, which has no reference speaker. flashder returns a JER of 100 percent for such a recording and excludes it from the overall JER, as dscore does.

## 7. Speaker mapping for JER

dscore maps speakers so as to minimise the sum of the per-speaker Jaccard errors. pyannote.metrics uses the mapping that maximises co-occurrence time, the same as for DER.

```
ref  A [0, 1]
     B [0, 3]
hyp  x [0, 8]
     y [2, 3]
uem  [0, 8]
```

| tool | JER |
|---|---|
| dscore | 77.08 |
| pyannote.metrics | 81.25 |

Mapping A to y and B to x gives 3 s of co-occurrence; mapping A to x and B to y gives 2 s. pyannote.metrics chooses the first: A to y has error 1, because A and y never co-occur, and B to x has error 1 - 3/8, which gives a JER of 81.25. dscore chooses the second: A to x has error 1 - 1/8 and B to y has error 1 - 1/3, which gives 77.08. The DER is 175.00 in all tools, because the DER mapping is the same.

pyannote.metrics also applies the collar and the overlap exclusion to JER. dscore computes JER on the whole UEM regardless of these options.

## 8. JER on 10 ms frames

dscore computes DER exactly, through md-eval, but computes JER on frames of 10 ms. Frame i covers time 0.01 i, and a speaker is active in frame i if 0.01 i lies in one of its turns. Boundaries that do not fall on the 10 ms grid are therefore rounded.

```
ref  A [0, 1.004]
hyp  x [0, 1.0]
uem  [0, 2]
```

| tool | DER | JER |
|---|---|---|
| md-eval | 0.40 | |
| dscore | 0.40 | 0.99 |
| pyannote.metrics | 0.40 | 0.40 |

The exact Jaccard error is 1 - 1/1.004 = 0.40 percent. On the frame grid, A covers 101 frames and x covers 100, which gives 1 - 100/101 = 0.99 percent. flashder uses frames for JER in the `dscore` style only; `jer_resolution=None` forces the exact value.

A further consequence of the frame grid is that a turn shorter than 10 ms may cover no frame. dscore still counts its speaker as a reference speaker, with a Jaccard error of 100 percent unless a hypothesis speaker also covers no frame. flashder reproduces this in the `dscore` style.

## 9. Other sources of small differences

- dscore writes the turns it passes to md-eval with three decimals. Times that are not multiples of 1 ms are rounded before scoring.
- md-eval prints durations with two or three decimals and the DER with two.
- When two speaker mappings have exactly the same co-occurrence time, each tool chooses one of them in its own way, and the confusion time can differ. This is rare with times in seconds and more frequent with frame-level input, where durations are integer numbers of frames.
- All three tools report the micro average over recordings: durations are summed over the data set before dividing. JER is averaged over all reference speakers in the data set. flashder also provides the unweighted mean over recordings with `average="macro"`.
