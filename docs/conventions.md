# Scoring conventions

NIST `md-eval-22.pl`, dscore and pyannote.metrics all compute the diarization error rate (DER), and dscore and pyannote.metrics also compute the Jaccard error rate (JER). They make different choices in several places, and the same input can receive different scores. FlashDER makes one choice at each of these places and applies it to both metrics. This document lists each difference with a small example, the value reported by each tool, the value reported by FlashDER and the reason for the FlashDER choice.

The values were obtained with `md-eval-22.pl` (version 22, as distributed with dscore), dscore at commit `e02f949` and pyannote.metrics 4.0.0. Times are in seconds and scores in percent. Every example is reproduced in `tests/test_conventions.py`, which checks the FlashDER values and, when the tools are installed, the values of the original tools.

The `style` argument (`md-eval`, `dscore` or `pyannote`) reproduces the choices of one tool, for comparison with results published with that tool. Without `style`, FlashDER uses the conventions described here, and the FlashDER rows of the tables below give the values it reports in that case.

In the examples, `ref` and `hyp` list turns as `speaker [start, end]`. md-eval does not compute JER, and its JER cells are marked n.a.

## 1. Collar width

md-eval (`-c`) and dscore (`--collar`) exclude a region of the given length on each side of every reference boundary. pyannote.metrics (`collar=`) takes the total width of that region, half on each side. FlashDER (`collar=`) takes the length on each side.

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
| FlashDER `collar=0.25` | 0.00 |

With pyannote `collar=0.25` the excluded region around time 0 is [-0.125, 0.125], and [0.125, 0.2] is counted as missed speech. The collar is a tolerance on the position of a reference boundary, and the tolerance is the distance from the boundary within which a hypothesis boundary is accepted. FlashDER states it as that distance, as in the NIST evaluation plans and in md-eval: `collar=0.25` corresponds to pyannote `collar=0.5`.

## 2. Default UEM

When no UEM is given, md-eval scores the extent of the reference, from the start of the first reference turn to the end of the last one. dscore and pyannote.metrics score the extent of the reference and hypothesis together. FlashDER scores the extent of the reference and hypothesis together.

```
ref  A [0, 10]
hyp  x [0, 12]
no UEM
```

| tool | DER | JER |
|---|---|---|
| md-eval | 0.00 | n.a. |
| dscore | 20.00 | 16.67 |
| pyannote.metrics | 20.00 | 16.67 |
| FlashDER | 20.00 | 16.67 |

The hypothesis speech in [10, 12] is outside the region scored by md-eval and is not counted. The region scored by md-eval depends on the reference alone, so hypothesis speech before the first reference turn or after the last one is never a false alarm. A system that marks the whole recording as speech is not penalised for the part outside the reference extent, and a recording whose reference is a single short turn has a DER close to zero for a hypothesis that is wrong almost everywhere. FlashDER scores every instant at which the reference or the hypothesis has speech, so every false alarm is counted. When part of a recording is not to be scored, this is stated with a UEM.

## 3. Region used for the speaker mapping

md-eval and dscore compute the optimal mapping between reference and hypothesis speakers on the whole UEM, before collars and overlap regions are removed. pyannote.metrics computes it on the scored region only. The two can produce different mappings when collars or overlap exclusion remove most of a speaker's speech. FlashDER computes the mapping on the scored region.

```
ref  A [0, 1] [2, 3] [4, 5] [6, 7]
     B [8, 10]
hyp  x [0, 1] [2, 3] [4, 5] [6, 7] [8, 10]
uem  [0, 10]
collar 0.5 s per side
```

| tool | DER | JER |
|---|---|---|
| md-eval `-c 0.5` | 100.00 | n.a. |
| dscore `--collar 0.5` | 100.00 | 66.67 |
| pyannote `collar=1.0` | 0.00 | 0.00 |
| FlashDER `collar=0.5` | 0.00 | 0.00 |

On the whole UEM, x shares 4 s of speech with A and 2 s with B, so md-eval maps x to A. The collars remove all of A's turns and leave [8.5, 9.5], where only B speaks. md-eval counts this second as confusion. pyannote.metrics maps x to B, the only speaker in the scored region, and reports no error.

The DER is defined as the error under the one to one speaker mapping that minimises it. A mapping computed on the whole UEM minimises the confusion on a region that is partly not scored, and the mapping it selects can be far from optimal on the scored region: in this example the hypothesis is correct at every scored instant and md-eval reports a DER of 100 percent. FlashDER computes the mapping on the scored region, so the reported DER is the minimum over mappings of the scored error, and the collar and the overlap exclusion are applied once, before any quantity is computed. In this example A has no speech in the scored region and is not a scored speaker, B is mapped to x, and FlashDER reports no error for DER and JER.

## 4. Turns cut by the UEM

dscore crops all turns to the UEM before calling md-eval. A turn that extends past the end of a UEM region therefore ends at the UEM boundary, and md-eval places a collar there. md-eval and pyannote.metrics, given the uncropped turns, place no collar at that point. FlashDER places no collar at a UEM boundary.

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
| FlashDER `collar=0.25` | 4.21 |

md-eval scores [0.25, 5] and counts [4.8, 5] as missed (0.2 / 4.75). dscore also excludes [4.75, 5] and reports no error.

The collar accounts for the uncertainty in the position of a reference speaker boundary. A UEM boundary is not a speaker boundary: the reference speech continues past it, and its position is chosen by the evaluator, not by the annotator of the speech. A collar at a UEM boundary removes one collar width of valid scored time at each end of every UEM region, and a hypothesis that stops short of the UEM boundary, as in this example, is not penalised. FlashDER applies collars at reference turn boundaries only.

## 5. Overlapping turns of the same speaker

Reference files sometimes contain two turns of the same speaker that overlap in time. The tools treat them in three different ways.

- md-eval counts the speaker once when computing errors, but counts both turns when it decides where reference overlap occurs (`-1`).
- dscore merges the turns before scoring.
- pyannote.metrics counts the speaker once per turn, both for errors and for overlap.

FlashDER counts the speaker once, for errors and for overlap, which is the same as merging the turns.

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
| FlashDER | 16.67 | 16.67 |

Without overlap exclusion, md-eval and dscore score 3 s of reference speech with 0.5 s missed. pyannote.metrics counts A twice in [1, 2], which gives 4 s of reference speech and 1.5 s missed. With overlap exclusion, md-eval and pyannote.metrics treat [1, 2] as overlap and remove it (0.5 / 2). dscore has merged the turns, finds no overlap and scores all 3 s (0.5 / 3).

A speaker cannot speak twice at the same time. Two overlapping turns of one speaker come from the way the speech was segmented, for example from an annotation at two granularities or from a correction, and not from the speech itself. The score should not depend on how the speech of a speaker is split into turns: splitting a turn in two, or merging two adjacent or overlapping turns of the same speaker, must leave DER and JER unchanged. FlashDER therefore treats a speaker as active or inactive at each instant, counts the speaker once in the errors, and defines overlap as a region where two or more distinct speakers are active. The `md-eval` and `pyannote` styles reproduce the values of those tools.

Hypothesis turns are subject to the same difference: pyannote.metrics counts overlapping turns of one hypothesis speaker more than once, which is the cause of the difference in DER between pyannote.metrics and the other tools in the benchmarks of the README. FlashDER counts a hypothesis speaker once as well.

## 6. Recordings without reference speech

md-eval only scores recordings that appear in the reference. dscore reports a DER of 100 percent for such a recording when the hypothesis contains speech, but its overall DER is taken from md-eval and excludes it. pyannote.metrics scores every recording passed to it. FlashDER scores every recording of the UEM or, when no UEM is given, every recording that appears in the reference or in the hypothesis.

```
recording 1   ref A [0, 10]   hyp x [0, 10]   uem [0, 10]
recording 2   no reference    hyp y [0, 5]    uem [0, 5]
```

| tool | overall DER |
|---|---|
| md-eval | 0.00 |
| dscore | 0.00 |
| pyannote.metrics | 50.00 |
| FlashDER | 50.00 |

pyannote.metrics adds 5 s of false alarm to 10 s of reference speech. md-eval and dscore drop the hypothesis of recording 2, and the 5 s of false alarm are not counted, which is the same omission as in point 2. Speech that the system assigns to a recording in which the reference has none is an error of the system, and whether the recording is to be scored at all is stated by the UEM. FlashDER counts this speech as false alarm.

The JER of a recording is the mean of the Jaccard error over its reference speakers and is undefined when there are none. The `JaccardErrorRate` of pyannote.metrics raises `ZeroDivisionError` on recording 2. FlashDER reports a JER of 100 percent for such a recording. The overall JER is the mean over all reference speakers in the data set, to which the recording contributes no speaker, as in dscore.

## 7. Speaker mapping for JER

dscore maps speakers so as to minimise the sum of the Jaccard errors over speakers. pyannote.metrics uses the mapping that maximises the shared speech time, the same as for DER. FlashDER uses the mapping that minimises the sum of the Jaccard errors.

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
| FlashDER | 77.08 |

Mapping A to y and B to x gives 3 s of shared speech; mapping A to x and B to y gives 2 s. pyannote.metrics chooses the first: A to y has error 1, because A and y never speak at the same time, and B to x has error 1 - 3/8, which gives a JER of 81.25. dscore chooses the second: A to x has error 1 - 1/8 and B to y has error 1 - 1/3, which gives 77.08. The DER is 175.00 in all tools, because the DER mapping is the same.

Each metric is defined as the minimum over one to one speaker mappings of its own cost. The DER mapping maximises the shared speech time and does not minimise the Jaccard error, so the JER of pyannote.metrics is not the minimum over mappings, and it depends on which of several DER mappings with equal shared time is selected. FlashDER computes the JER mapping from the matrix of Jaccard errors, as in the definition of the DIHARD II challenge.

pyannote.metrics also applies the collar and the overlap exclusion to JER. dscore computes JER on the whole UEM regardless of these options. FlashDER computes JER on the scored region, the same region as DER. The collar and the overlap exclusion define which part of the recording is scored, and the two metrics reported for one scoring run describe the same part of the data. The `dscore` style reproduces the dscore values.

## 8. JER on 10 ms frames

dscore computes DER exactly, through md-eval, but computes JER on frames of 10 ms. Frame i covers time 0.01 i, and a speaker is active in frame i if 0.01 i lies in one of its turns. Boundaries that do not fall on the 10 ms grid are therefore rounded. FlashDER computes both metrics exactly.

```
ref  A [0, 1.004]
hyp  x [0, 1.0]
uem  [0, 2]
```

| tool | DER | JER |
|---|---|---|
| md-eval | 0.40 | n.a. |
| dscore | 0.40 | 0.99 |
| pyannote.metrics | 0.40 | 0.40 |
| FlashDER | 0.40 | 0.40 |

The exact Jaccard error is 1 - 1/1.004 = 0.40 percent. On the frame grid, A covers 101 frames and x covers 100, which gives 1 - 100/101 = 0.99 percent. A frame grid moves every boundary by up to one frame, and dscore computes its two metrics from two different versions of the same data, one exact and one rounded. FlashDER computes DER and JER in continuous time from the same set of intervals. The `resolution` argument puts both metrics on a common grid when frame level scoring is wanted, and the `dscore` style reproduces the dscore value.

A further consequence of the frame grid is that a turn shorter than 10 ms may cover no frame. dscore still counts its speaker as a reference speaker, with a Jaccard error of 100 percent unless a hypothesis speaker also covers no frame. In exact scoring a turn of any positive length is counted with its length, and a reference speaker is scored whenever part of its speech lies in the scored region.

## 9. Other sources of small differences

- dscore writes the turns it passes to md-eval with three decimals. Times that are not multiples of 1 ms are rounded before scoring. FlashDER scores the times as given.
- md-eval prints durations with two or three decimals and the DER with two. FlashDER returns the error components in seconds and the rates as fractions, without rounding.
- When two speaker mappings have exactly the same shared speech time, each tool chooses one of them in its own way, and the confusion time can differ. This is rare with times in seconds and more frequent with frame level input, where durations are integer numbers of frames.
- All three tools report the micro average over recordings: durations are summed over the data set before dividing. JER is averaged over all reference speakers in the data set. FlashDER reports the same averages and also provides the unweighted mean over recordings with `average="macro"`.
