# FlashDER

FlashDER computes the diarization error rate (DER) and the Jaccard error rate (JER) for speaker diarization. It gives the same results as NIST [`md-eval-22.pl`][md-eval], [dscore][dscore] and [pyannote.metrics][pyannote]. On the benchmarks below, scoring from Python is 45 to 390 times faster than pyannote.metrics, and the command line is 2 to 13 times faster than `md-eval-22.pl` and dscore.

Each recording is cut at every reference, hypothesis, UEM and collar boundary. The resulting intervals are the columns of two boolean matrices, one for the reference speakers and one for the hypothesis speakers. Missed speech, false alarm and speaker confusion are then counts over these matrices, weighted by interval duration. The speaker mapping is found with the Hungarian algorithm on the matrix of co-occurrence times. Scoring is exact in continuous time. A fixed frame grid can be used instead, and a [PyTorch][pytorch] implementation scores batches of frame-level activities on CPU or GPU.

## Installation

From a clone of this repository:

```sh
pip install .            # numpy and scipy only
pip install ".[torch]"   # adds the PyTorch backend
pip install -e ".[test]" # development install with test dependencies
```

Python 3.9 or later is required. The Python package and the command are named `flashder`.

## Command line

```sh
flashder -r examples/ref.rttm -s examples/sys.rttm -u examples/all.uem -c 0.25 -1 --per-file
```

```
File               DER  Miss    FA   Conf    JER  Scored(s)
-----------------------------------------------------------
rec1             25.97  0.00  0.00  25.97  44.26       7.70
rec2             26.19  0.00  2.38  23.81  41.92      10.50
-----------------------------------------------------------
*** OVERALL ***  26.10  0.00  1.37  24.73  43.09      18.20
```

Values are percentages. `Scored(s)` is the scored reference speaker time in seconds. `md-eval-22.pl -c 0.25 -1` reports the same overall DER of 26.10 percent for these files. JER ignores the collar and the overlap exclusion, as in dscore.

Options:

| option | meaning |
|---|---|
| `-r`, `-s` | reference and system files (RTTM, SegLST, STM or lab, one or more each) |
| `-R`, `-S` | text files listing reference or system files, one per line |
| `-u` | UEM file |
| `-c` | collar in seconds on each side of every reference boundary |
| `-1` | exclude regions where the reference has overlapping speech |
| `--style` | `md-eval` (default), `dscore` or `pyannote` |
| `--average` | `micro` (default) or `macro` |
| `--resolution` | score on frames of this size in seconds instead of exactly |
| `--no-jer` | compute DER only |
| `--json` | print JSON |

`python -m flashder` is equivalent to `flashder`.

## Python

```python
import flashder

report = flashder.score("ref.rttm", "sys.rttm", "all.uem", collar=0.25, skip_overlap=True)
report.der                 # overall DER, as a fraction
report.jer                 # overall JER
report.files["rec1"].der   # DER of one recording
report.overall("macro")    # macro averages from the same report
print(report.table())
```

Reference and hypothesis may be file paths, a dictionary mapping recording identifiers to `flashder.Segments`, a single `Segments` object, or pyannote `Annotation` objects. `Segments` holds three arrays: start times, end times and speaker labels.

```python
from flashder import Segments, score_file

ref = Segments.from_tuples([(0.0, 5.2, "A"), (4.8, 9.0, "B")])
hyp = Segments.from_tuples([(0.0, 5.0, "s1"), (5.0, 9.0, "s2")])
der, jer = score_file(ref, hyp, uem=[[0.0, 9.0]], collar=0.25)
der.miss, der.false_alarm, der.confusion, der.total  # seconds
der.mapping                                          # {"A": "s1", "B": "s2"}
```

### Frame-level input

Frame-level activities, for example the output of an end-to-end diarization model, can be scored without conversion to segments.

```python
from flashder import der_frames
d = der_frames(ref_frames, hyp_frames, frame_shift=0.01, collar=25)  # arrays of shape (frames, speakers)
```

The PyTorch backend scores a batch at once:

```python
from flashder.torch import der_frames, jer_frames

out = der_frames(ref, hyp_probs, mask=valid, collar=25)  # (batch, frames, speakers)
der = out["errors"].sum() / out["total"].sum()            # micro average over the batch
```

Floating-point hypotheses are thresholded at `threshold=0.5`. The collar is given in frames on each side. With up to 8 speakers the optimal mapping is found by exhaustive search over permutations on the device. With more speakers, scipy's Hungarian solver is used on the CPU.

## Scoring conventions

The reference tools differ in several details. The `style` argument selects which tool to reproduce. [docs/conventions.md](docs/conventions.md) describes each difference with an example and the value reported by each tool.

| | `md-eval` (default) | `dscore` | `pyannote` |
|---|---|---|---|
| UEM when none is given | extent of the reference | extent of reference and hypothesis | extent of reference and hypothesis |
| region used for the speaker mapping | whole UEM | whole UEM | scored region only |
| turns cropped to the UEM and merged before collars are computed | no | yes | no |
| JER speaker mapping | minimum Jaccard error | minimum Jaccard error | maximum overlap |
| JER region | UEM | UEM | scored region |
| JER computed on 10 ms frames | no | yes | no |
| recordings absent from the reference | ignored | ignored | counted as false alarm |

Further notes:

- The collar is given in seconds on each side of a reference boundary, as in md-eval (`-c`) and dscore (`--collar`). pyannote.metrics uses the total width, so pyannote `collar=0.5` corresponds to `collar=0.25` here.
- `skip_overlap=True` excludes regions where two or more reference segments are active. This is md-eval `-1`, dscore `--ignore_overlaps` and pyannote `skip_overlap=True`.
- `average="micro"` (default) sums durations over all recordings before dividing, which is what md-eval, dscore and pyannote.metrics report. JER is then the mean over all reference speakers in the data set. `average="macro"` is the unweighted mean of per-recording values.
- A speaker who has several overlapping turns at the same time is counted once, as in md-eval and dscore. pyannote.metrics counts such a speaker once per turn. `count_overlapping_turns=True` reproduces this behaviour.
- When two speaker mappings have exactly the same co-occurrence time, the tools may choose different ones and report different confusion. This mostly occurs with frame-level input, where durations are integer numbers of frames.

## File formats

| format | extension | notes |
|---|---|---|
| [RTTM][rttm] | `.rttm` | `SPEAKER` lines; other line types are skipped |
| [SegLST][seglst] | `.json` | CHiME-7/8 and [MeetEval][meeteval]; uses `session_id`, `start_time`, `end_time`, `speaker` |
| [STM][stm] | `.stm` | `inter_segment_gap` and `IGNORE_TIME_SEGMENT_IN_SCORING` lines are skipped |
| lab | `.lab`, `.txt` | `start end speaker` per line ([Audacity labels][audacity]); the file name is the recording identifier |
| [Kaldi][kaldi] | | `segments` and `utt2spk`, read with `flashder.io.load_kaldi` |
| UEM | `.uem` | `file channel start end` |

The format is inferred from the extension or, failing that, from the file content. `flashder.io` also writes RTTM, SegLST, STM, lab and UEM files.

## Agreement with other tools

The test suite compares FlashDER with the reference tools on random data with overlapping speech, several recordings and UEMs with gaps. Each comparison is repeated with and without collar and with and without overlap exclusion.

| tool | quantities compared | tolerance |
|---|---|---|
| `md-eval-22.pl` | scored, missed, false alarm and confusion time; DER | printed precision (0.01) |
| dscore | DER and JER, per recording and overall | 1e-4 percent |
| pyannote.metrics 4.0 | DER components and JER | 1e-9 |

The PyTorch backend is compared with the numpy implementation on CPU and, when available, CUDA and MPS devices.

## Speed

Time to score two synthetic data sets on an Apple M5 (Python 3.13, numpy 2.3, scipy 1.16, Perl 5.34). Library timings exclude file reading. md-eval and dscore can only be run as scripts, so they are timed as whole processes; the FlashDER command line is timed in the same way for comparison. All tools report the same DER, with two exceptions. pyannote.metrics counts overlapping turns of the same hypothesis speaker more than once (see above). simpleder scores one recording at a time and has no UEM; the benchmark averages its per-recording values weighted by reference duration, which differs slightly from the other tools when a collar is used.

500 recordings of 5 minutes, 2 to 4 speakers, 39,139 reference turns:

| tool | DER | DER, collar 0.25 s, overlap excluded | DER and JER |
|---|---|---|---|
| FlashDER | 48 ms | 69 ms | 65 ms |
| [spyder][spyder] 0.4.1 | 83 ms | 85 ms | no JER |
| [simpleder][simpleder] 0.0.8 | 889 ms | 688 ms | no JER |
| [pyannote.metrics][pyannote] 4.0.0 | 4.06 s | 3.13 s | 4.93 s |
| FlashDER command line | 0.29 s | 0.29 s | 0.31 s |
| [md-eval-22.pl][md-eval] | 1.32 s | 1.37 s | no JER |
| [dscore][dscore] | 3.73 s (1) | 3.71 s (1) | 3.74 s |

24 recordings of 30 minutes, 4 to 5 speakers, 12,257 reference turns:

| tool | DER | DER, collar 0.25 s, overlap excluded | DER and JER |
|---|---|---|---|
| FlashDER | 8.9 ms | 14 ms | 10 ms |
| spyder 0.4.1 | 27 ms | 28 ms | no JER |
| simpleder 0.0.8 | 1.51 s | 0.99 s | no JER |
| pyannote.metrics 4.0.0 | 3.23 s | 2.61 s | 4.02 s |
| FlashDER command line | 0.21 s | 0.21 s | 0.21 s |
| md-eval-22.pl | 0.44 s | 0.47 s | no JER |
| dscore | 1.25 s (1) | 1.30 s (1) | 1.27 s |

(1) dscore always computes JER and its clustering metrics together with DER, so these times include them.

About 0.16 s of the command-line time is the import of numpy and scipy.

Frame-level DER on a batch of 64 chunks of 5000 frames with 4 speakers and a 25-frame collar:

| implementation | time |
|---|---|
| `flashder.torch`, MPS | 3.1 ms |
| `flashder.torch`, CPU | 7.0 ms |
| `flashder.der_frames`, loop over the batch | 16.5 ms |
| spyder, loop over the batch | 14.1 ms |
| pyannote.metrics, loop over the batch | 180 ms |

To reproduce:

```sh
pip install pyannote.metrics spy-der simpleder intervaltree tabulate
git clone https://github.com/nryant/dscore.git
python benchmarks/bench.py --dscore-dir dscore
```

## Tests

```sh
pip install -e ".[test,torch]"
pytest
```

The comparisons with md-eval and dscore clone dscore, which includes `md-eval-22.pl`, into `tests/.cache`. Set `DSCORE_DIR` to use an existing copy. These tests are skipped when Git, Perl or network access is unavailable.

## Citation

If you use FlashDER, please cite it as follows.

```bibtex
@software{cornell2026flashder,
  author  = {Cornell, Samuele},
  title   = {{FlashDER}: fast and exact diarization error rate and Jaccard error rate},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/popcornell/FlashDER}
}
```

The metadata is also in `CITATION.cff`. DER is defined in the [NIST RT-09 evaluation plan][rt09], and JER in the [DIHARD II challenge][dihard2] ([Ryant et al., 2019][dihard2-paper]).

## Related tools

- [md-eval-22.pl][md-eval]: NIST scoring script, distributed with dscore
- [dscore][dscore]: DIHARD scoring tool (DER through md-eval, JER and clustering metrics)
- [pyannote.metrics][pyannote]: evaluation metrics of the pyannote toolkit
- [spyder][spyder]: DER in C++ with Python bindings
- [simpleder][simpleder]: DER in Python
- [MeetEval][meeteval]: meeting transcription evaluation, with DER through md-eval and dscore

## License

MIT

[md-eval]: https://github.com/nryant/dscore/blob/master/scorelib/md-eval-22.pl
[dscore]: https://github.com/nryant/dscore
[pyannote]: https://github.com/pyannote/pyannote-metrics
[spyder]: https://github.com/desh2608/spyder
[simpleder]: https://github.com/wq2012/SimpleDER
[meeteval]: https://github.com/fgnt/meeteval
[pytorch]: https://pytorch.org
[rttm]: https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf
[rt09]: https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf
[seglst]: https://github.com/fgnt/meeteval#segment-wise-long-form-speech-transcription-annotation-seglst
[stm]: https://github.com/usnistgov/SCTK/blob/master/doc/infmts.htm#stm_fmt_name_0
[audacity]: https://manual.audacityteam.org/man/importing_and_exporting_labels.html
[kaldi]: https://kaldi-asr.org/doc/data_prep.html
[dihard2]: https://dihardchallenge.github.io/dihard2/
[dihard2-paper]: https://arxiv.org/abs/1906.07839
