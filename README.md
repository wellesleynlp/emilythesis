## A Computational Approach to Foreign Accent Classification
#### _Emily Ahn_
##### Wellesley College Undergraduate Senior Thesis // May 2016

Full write-up in pdf form can be found in the `writeup/` folder or in the [Wellesley College repositories](http://repository.wellesley.edu/thesiscollection/323/ "Wellesley College archives").

---

### 1. Folders in this repository:

#### `alignments/`
- Contains forced alignents and transcriptions for the text-dependent classifier
- The 7 subfolders correspond to ach of the 7 transcribed accents
- Simple transcriptions of speech files are organized by accent. The format is a `.csv` file compiled via releasing transcription tasks on Amazon Mechanical Turk, then personally cleaned up by the author. _*Note: errors still exist in some transcriptions._

#### `trans-results/` and `untrans-results/`
- Console print logs of results from the text-dependent "trans" (transcribed) classifier and the text-independent "untrans" (untranscribed) classifier

#### `traintestsplit/`
- Contains lists of filenames that were split into train and test data, via a randomized 75:25 split

#### `formants/`
- Script and data (in csv format) to test GMM classifcation based on 3 vowel formants for AR, CZ, and IN accents

---

### 2. Main scripts
- Text-independent (untranscribed) Classifier
    - `gmmClassifier.py` || full script; loads data, trains models, classify test data
    - `gmmTrain.py` || modularizes training only, stores models in directory
    - `gmmTest.py` || modularizes testing only
- Text-dependent (transcribed) Classifier
    - `parseTextGrid.py` || prepares data by converting forced alignments of speech into plp features (sorted by accent and phoneme)
    - `phoneClassifier.py` || full script; gmm Classification of transcribed phonemes

### 3. Miscellaneous scripts
- `avgnpy_test.py` || takes average of each dimension of PLP vector across all time windows from a given sound file
- `miniClassifier.py` || does univariate GMM classification of AR, HI, MA


