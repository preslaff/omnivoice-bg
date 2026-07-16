# OmniVoice-BG Pipeline Guide

Operational runbook for the whole Bulgarian stress + TTS pipeline: what each
stage does, the exact commands, the expected outputs, and the order to execute
them **after the parliament crawl finishes**. Companion documents:

- [OMNIVOICE_BG.md](../OMNIVOICE_BG.md) — the serving layer (server, engines, TrainGen client).
- [BG_FINETUNE_PLAN.md](../BG_FINETUNE_PLAN.md) — OmniVoice fine-tuning deep-dive.
- [BG_PARLIAMENT_INGESTION.md](BG_PARLIAMENT_INGESTION.md) — crawl internals + gap table.
- `interspeech2023.pdf` — the Accentor paper we are reproducing (Geneva et al.).

---

## The big picture

Two coupled pipelines. **Serving** (works today) and **Accentor reproduction**
(replaces the dictionary front-end with a contextual one):

```
SERVING (today)
  plain BG text ──▶ bg_accent (rechko lookup + overrides) ──▶ marked text
              ──▶ OmniVoice-BG (fine-tuned, cloned voice) ──▶ 24 kHz WAV
                                   ▲
                        serve.py :8002  ◀── TrainGen POST /synth

ACCENTOR REPRODUCTION (this guide)
  rechko dump ─▶ stress-candidates DB ─▶ phonetic lexicon (stressed/unstressed vowels)
                       │                        │
  chitanka books ─▶ text-only marking ─▶ Accentor PRETRAIN (done, SER 0.003%)
                       │                        │
  parliament crawl ─┐  │                        ▼
  real audiobooks ──┴─▶ Kaldi alignment ─▶ per-token stress labels (human audio)
                                                │
                                                ▼
                              Accentor FINE-TUNE (homographs) ─▶ replaces rechko
                                                │                 in serve.py
                                                ▼
                              re-mark finetune data ─▶ OmniVoice-BG re-finetune
```

Why: rechko is a word→one-stress lookup. It cannot resolve homographs
(въ́лна/вълна́, че́та/чета́ — 1 in 23 BG words, 34% of utterances per the paper).
The Accentor reads the whole sentence. Its training labels for ambiguous words
can only come from **human speech**, which is what the alignment branch produces.

### Three models, and which ones are acoustic (paper-critical distinction)

A recurring confusion worth stating explicitly: **the Accentor is not an
acoustic model.** The system is a relay of three specialists, and only the two
ends touch audio:

```
Kaldi AM  ──hears stress in human speech──▶  per-token stress labels
(analysis; a1/a0 phones)                          │
                                                  ▼
Accentor  ◀──trained on those labels──  learns WHERE stress goes (TEXT→TEXT)
(placement; no audio, ever)                       │
                                                  ▼  marks plain text w/ U+0301
OmniVoice-BG  ──learns to VOICE the mark──▶  audio with correct stress
(synthesis; the generator TrainGen calls)
```

1. **Kaldi acoustic model (analysis side)** — the only model that learns *how
   a stressed vowel sounds*, because we give it separate phones for stressed
   vs unstressed vowels (`a1`/`a0`, `o1`/`o0`, …); during alignment it learns
   the acoustic signature (duration, F0, energy) well enough to DECODE which
   vowel was stressed from audio. This is `tri3b`/`tri3b_comb` (pilot) → the
   annotation TDNN (milestone A). A recognizer, not a generator: it hears
   stress, does not produce it. **This model manufactures the training labels.**
2. **Accentor (placement)** — reads characters, emits characters (plain text →
   U+0301-marked text). Never sees a waveform. Learns only *where* stress
   falls from surrounding words (чета́ книга vs въоръжена че́та). Its speech-
   derived labels come FROM model #1; the Accentor itself is text-only.
3. **OmniVoice-BG (synthesis side)** — learns to RENDER the U+0301 mark as the
   acoustic correlates of stress (stage 6f re-finetune). The model that
   produces the final TrainGen audio.

Why decouple #2 and #3 instead of one text→stressed-audio model: stress
*placement* (a hard, context-dependent linguistic decision, trainable on
billions of cheap text tokens) and stress *acoustics* (a rendering problem
needing aligned audio) are different skills with different data economics.
Separating them — the paper's core thesis — yields contextual homograph
correctness AND OmniVoice's zero-shot cloning simultaneously.

---

## Status snapshot (2026-07-04)

| Stage | State |
|---|---|
| 1. Lexicon (candidates DB + phonetic lexicon) | **DONE** — 84,456 ambiguous forms; 2.53M pronunciations; 34 vowel phones (a0/a1…) + 22 consonants |
| 2. Chitanka text corpus + marking | **DONE** (run 3: 6.0M sentences, 71.5M tokens, 78% labeled) — **re-mark pending** (и́/му́ precedence fix landed after run 3) |
| 3. Accentor text-only pretrain | **DONE** — best SER 0.003% @ step 84K; battery 11/25 (expected baseline) — **retrain pending** after re-mark |
| 4. Parliament crawl | **RUNNING** — ~100 h banked; milestone A ≈ 500 h raw; paper parity ≈ 3,900 h raw |
| 5. Kaldi alignment (audiobook pilot) | **DONE** — 63 h aligned (tri3b SAT), stress choices harvested; native vetting pending |
| 5b. Parliament viability pilot (banked ~100 h) | **READY** — `kaldi_parliament_pilot.sh` (segmentation → alignment → labels) |
| 6. Parliament segmentation + annotation TDNN | **NOT STARTED** — gated on milestone A; 6a is the scale-up of 5b |
| 7. Accentor fine-tune on speech labels | **NOT STARTED** — gated on 5-vetting / 6 |
| 8. Serving + TrainGen | **WORKS** — `serve_bg.sh`, rechko front-end; Accentor swap-in is the end goal |

Machine constraints (preslaff-gpu): 60 GB RAM, **8 GB swap**, 24 cores, 16 GB
GPU, ~1.5 TB free NVMe shared with the crawl. See Pitfalls before running
anything with `nj=$(nproc)`.

---

## Stage 1 — Stress lexicon

Build once; rebuild only if rechko data or the battery changes.

```bash
# 1a. rechko dump -> one-stress sqlite (serving lexicon)
./venv/bin/python build_bg_lexicon.py --dump ../rechko/db.sql \
    --out data/bg_lexicon/bg_stress.sqlite

# 1b. multi-stress CANDIDATES db (recall-oriented: every plausible stress per form)
./venv/bin/python build_stress_candidates.py \
    --src data/bg_lexicon/bg_stress.sqlite \
    --battery config/bg/homograph_battery.json \
    --out data/bg_lexicon/bg_stress_candidates.sqlite --validate

# 1c. phonetic lexicon: candidates -> stressed/unstressed vowel phones
#     (a1 = stressed 'а', a0 = unstressed; consonants unnumbered)
./venv/bin/python build_lexicon_phones.py \
    --cand data/bg_lexicon/bg_stress_candidates.sqlite \
    --out-lexicon data/bg_lexicon/lexicon_full.txt \
    --out-phones data/bg_lexicon/phones.txt

# coverage check against a corpus (optional)
./venv/bin/python lexicon_coverage.py
```

Outputs: `bg_stress_candidates.sqlite` (84,456 ambiguous forms),
`lexicon_full.txt` (~2.53M forms / 3.35M pronunciations), `phones.txt`.

**Curation inputs** (native-speaker maintained):
`config/bg/homograph_battery.json` (confirmed/rejected homographs),
`config/bg/function_words_unstressed.json` (43 clitics, zero-labeled),
`config/bg/lexicon_supplement.json`. The battery's `rejected` section documents
false homographs (замък, тела) — deletions applied directly to the DB.

---

## Stage 2 — Text corpus & marking (chitanka)

```bash
# 2a. clean the archive: headers/footers/credits stripped, sentences split
./venv/bin/python prep_chitanka.py --archive data/chitanka_archive --out data/chitanka
# -> data/chitanka/clean/*.txt + books.jsonl (8,002 books)

# 2b. text-only Accentor rows: unambiguous words marked from the lexicon,
#     function words zero-labeled, ambiguous+OOV masked from the loss
./venv/bin/python mark_chitanka_text.py \
    --corpus data/chitanka --sqlite data/bg_lexicon/bg_stress_candidates.sqlite \
    --out data/chitanka/marked.jsonl --max-tokens 80000000
# ~8 min. Healthy run: ~78% labeled (≈49% marked + ≈29% function),
# ~14% masked-ambiguous, ~8% masked-OOV.
```

> **PENDING:** re-run 2b once — the function-word precedence fix (rechko's
> marked clitic entries и́/му́ used to outrank the curated function list) landed
> after the current `marked.jsonl` was produced.

Design note: this replaces the abandoned synthetic-audio labeling. BgTTS-38M
was tested as a stress oracle and **failed systematically on homographs**
(always the frequency-prior reading, e.g. чета́ for че́та) — synthetic audio must
never be used to label ambiguous forms.

---

## Stage 3 — Accentor text-only pretrain

```bash
# training (≈4.6 h on the 16 GB GPU, batch 256, 100K steps)
./venv/bin/python train_accentor.py \
    --data data/chitanka/marked.jsonl --out exp/accentor_text
# monitor:
./venv/bin/tensorboard --logdir exp/accentor_text/tensorboard --port 6006

# evaluation: hard words (must be ALL correct) + battery (baseline only)
./venv/bin/python eval_accentor.py --ckpt exp/accentor_text/best.pt \
    --sentences battery_sentences.txt \
    --battery config/bg/homograph_battery_sentences.json
```

Architecture (accentor_model.py, paper §4): char embedding + caps feature +
6 FFT blocks (256-dim, 4 heads×128) + conv predictor, per-vowel BCE;
`--separable` gives the paper's 6.7M-param variant. Checkpoints: `last.pt`
(resume) and `best.pt` (lowest val SER). Loss masking: `mask_words` spans
(ambiguous/OOV) are excluded; function words supervise toward zero.

Result to beat next time: **val SER 0.003%** on single-stress words; battery
11/25 (the text-only model can only learn one reading per form — expected).

### Evaluation metrics (paper-ready definitions)

The model emits one probability per vowel position — P(this vowel carries the
word stress) — trained with per-vowel binary cross-entropy. All three eval
metrics are computed on the deterministic held-out split (rows whose
`crc32(id) % 100 < 1`, i.e. ~1%). Formal definitions, for the write-up:

- **`val_loss` — masked BCE (the soft optimization target).** The mean
  binary cross-entropy over *labeled* vowel positions only:
  `mean_{v ∈ L} [ −y_v·log σ(z_v) − (1−y_v)·log(1−σ(z_v)) ]`, where `z_v` is
  the logit, `y_v ∈ {0,1}` the label, and `L` the labeled vowels — stress-
  marked words (0/1 per vowel) and zero-label function words (all 0). Vowels
  in `mask_words` spans (ambiguous/OOV, whose truth is unknown at pretrain)
  are excluded from `L`. This is a continuous, calibration-sensitive measure
  (a correct-but-underconfident 0.9 still incurs loss) and the quantity
  gradient descent minimizes; it is the earliest and smoothest overfitting
  signal (it rises while train loss falls). NOT a stress-accuracy number.

- **`SER` — stress error rate (the headline metric; Geneva et al.'s
  reported measure).** Word-level accuracy of "is the stress on the correct
  vowel." Evaluated ONLY on *fully-labeled single-stress words*: content
  words where every vowel is labeled (none masked) and exactly one is truly
  stressed. For each such word take the model's **argmax** vowel
  `v* = argmax_v σ(z_v)` and count an error(venv) preslaff@preslaff-gpu:~/WORK/OmniVoice$ # battery — the controlled test
  ./venv/bin/python eval_accentor.py --ckpt exp/accentor_ft4/best.pt \
      --sentences battery_sentences.txt \
      --battery config/bg/homograph_battery_sentences.json

  # out-of-domain — generalization + носи/учи in the wild
  ./venv/bin/python eval_accentor_ood.py --ckpt exp/accentor_ft4/best.pt \
      --transcripts data/bg_parliament/transcripts --n 4000 --show 20
loaded exp/accentor_ft4/best.pt (step 5000, best SER 0.039%)

=== 1. Hard words (unambiguous — expect ALL correct) ===
  Наля́гането в систе́мата расте́ бъ́рзо.
    conf: Налягането:1.00 системата:1.00 расте:0.99 бързо:1.00
  Натисне́те буто́на и въведе́те паро́лата.
    conf: Натиснете:1.00 бутона:1.00 и:0.00 въведете:1.00 паролата:1.00
  Избере́те фа́йл от меню́то и отворе́те прозо́реца.
    conf: Изберете:1.00 файл:1.00 от:0.00 менюто:1.00 и:0.00 отворете:1.00 прозореца:1.00
  Запише́те про́мените, преди́ да затво́рите приложе́нието.
    conf: Запишете:1.00 промените:0.99 преди:1.00 да:0.00 затворите:1.00 приложението:1.00
  Свале́те архи́ва и разархивира́йте съдържа́нието.
    conf: Свалете:1.00 архива:1.00 и:0.00 разархивирайте:1.00 съдържанието:1.00

=== 2. battery_sentences.txt ===
  Пуло́верът е изплете́н от ме́ка о́вча вълна́.
    conf: Пуловерът:1.00 е:0.00 изплетен:1.00 от:0.00 мека:1.00 овча:1.00 вълна:0.96
  Огро́мна мо́рска вълна́ заля́ пла́жа.
    conf: Огромна:1.00 морска:1.00 вълна:0.78 заля:1.00 плажа:1.00
  Вся́ка ве́чер че́та по една́ глава́.
    conf: Всяка:1.00 вечер:1.00 чета:0.92 по:0.00 една:1.00 глава:1.00
  Че́та на Хри́сто Боте́в преми́на Дуна́ва.
    conf: Чета:0.61 на:0.00 Христо:0.62 Ботев:0.84 премина:1.00 Дунава:1.00
  Омогра́фите са ду́ми с разли́чно значе́ние.
    conf: Омографите:1.00 са:0.00 думи:1.00 различно:1.00 значение:1.00
  И аз му дума́м, и тя́ му ду́ма съ́щото.
    conf: И:0.00 аз:0.00 му:0.00 думам:0.87 и:0.00 тя:1.00 му:0.00 дума:1.00 същото:1.00
  На па́ркинга и́ма мно́го коли́.
    conf: На:0.00 паркинга:1.00 има:1.00 много:1.00 коли:1.00
  Купи́х паке́т печата́рски коли́.
    conf: Купих:1.00 пакет:1.00 печатарски:1.00 коли:1.00
  Изпи́х две́ коли́ в же́гата.
    conf: Изпих:1.00 две:1.00 коли:1.00 жегата:1.00
  Купи́х боя́ за сте́ните.
    conf: Купих:0.99 боя:1.00 за:0.00 стените:1.00
  Не се боя́ от ни́що.
    conf: Не:0.00 се:0.00 боя:1.00 от:0.00 нищо:1.00
  От ча́йника изли́за пара́.
    conf: От:0.00 чайника:1.00 излиза:1.00 пара:1.00
  Ня́мам ни́то една́ па́ра в джо́ба.
    conf: Нямам:1.00 нито:1.00 една:1.00 пара:0.97 джоба:1.00
  Ри́царите щурму́ваха ста́рия за́мък.
    conf: Рицарите:1.00 щурмуваха:1.00 стария:1.00 замък:1.00
  Клю́чът засе́дна в за́мъка на врата́та.
    conf: Ключът:1.00 заседна:1.00 замъка:1.00 на:0.00 вратата:1.00
  То́й гово́ри ти́хо по телефо́на.
    conf: Той:1.00 говори:0.82 тихо:1.00 по:0.00 телефона:1.00
  Говори́ по-висо́ко, не те чу́вам!
    conf: Говори:0.79 по-високо:1.00 не:0.00 те:0.00 чувам:1.00

=== 3. Homograph battery (masked in training — baseline, failures expected) ===
  [FAIL] вълна -> expect въ́лна (wool)
         пуло́верът е изплете́н от ме́ка о́вча вълна́.
  [PASS] вълна -> expect вълна́ (wave)
         огро́мна вълна́ заля́ ска́лите на брега́.
  [PASS] вълна -> expect вълна́ (wave (radio))
         радиоста́нцията излъ́чва на къ́са вълна́.
  [FAIL] пара -> expect па́ра (steam)
         от те́нджерата се вди́гаше гъ́ста пара́.
  [FAIL] пара -> expect пара́ (coin/cent)
         то́й ня́маше пу́кната па́ра в джо́ба си.
  [PASS] говори -> expect гово́ри ((he/she) speaks, 3sg)
         тя́ гово́ри ти́хо по телефо́на вся́ка ве́чер.
  [PASS] говори -> expect говори́ (speak!, imperative)
         говори́ по-висо́ко, защо́то не те чу́вам!
  [FAIL] чета -> expect че́та (band/detachment (noun; the paper's own example))
         хайду́шката чета́ преми́на та́йно през планина́та.
  [FAIL] чета -> expect чета́ (I read (verb, 1sg))
         вся́ка ве́чер че́та по една́ глава́ от кни́гата.
  [PASS] коса -> expect коса́ (hair)
         тя реши́ дъ́лгата си коса́ пред огледа́лото.
  [FAIL] коса -> expect ко́са (slanted (adj. fem.: коса черта))
         напи́ши адре́са с коса́ черта́ между́ две́те и́мена.
  [PASS] само -> expect са́мо (only (particle))
         и́скам са́мо една́ мину́та от вре́мето ти.
  [FAIL] само -> expect само́ (alone (neuter of сам))
         дете́то оста́на са́мо вкъ́щи ця́ла ве́чер.
  [PASS] моли -> expect мо́ли (begs (present 3sg))
         то́й мо́ли за про́шка все́ки път, кога́то сгреши́.
  [FAIL] моли -> expect моли́ (beg!, imperative)
         мо́ли се вре́мето да се задържи́ ху́баво.
  [PASS] носи -> expect но́си (carries (present 3sg))
         тя́ но́си те́жка ча́нта все́ки ден.
  [FAIL] носи -> expect носи́ (carry!, imperative)
         но́си бага́жа до кола́та, мо́ля те.
  [PASS] прави -> expect пра́ви (does (present 3sg))
         то́й пра́ви вси́чко възмо́жно, за да помо́гне.
  [FAIL] прави -> expect прави́ (do!, imperative)
         пра́ви какво́то ти ка́звам и ня́ма да съжаля́ваш.
  [PASS] купи -> expect ку́пи (bought (aorist 3sg))
         вче́ра тя́ ку́пи хля́б и мля́ко от магази́на.
  [PASS] купи -> expect купи́ (buy!, imperative)
         купи́ биле́ти за конце́рта още́ дне́с.
  [PASS] учи -> expect у́чи (studies (present 3sg))
         синъ́т им у́чи медици́на в со́фия.
  [FAIL] учи -> expect учи́ (study!, imperative)
         у́чи все́ки ден, ако́ и́скаш да изка́раш изпи́та.
  [PASS] ходи -> expect хо́ди (walks/goes (present 3sg))
         тя́ хо́ди на ра́бота пеша́ вся́ка су́трин.
  [FAIL] ходи -> expect ходи́ (walk!, imperative)
         хо́ди по-бъ́рзо, че ще закъсне́ем!

  battery: 13/25 (text-only baseline; fine-tune target is ~all)
loaded exp/accentor_ft4/best.pt (step 5000, train-SER 0.039%)
lexicon: 1,945,298 unambiguous, 84,441 ambiguous
sampled 4,000 parliament sentences from 74 files

=== automatic SER on parliament (out-of-domain) ===
  unambiguous words evaluated : 19,026
  stress error rate (SER)     : 0.021%
  mean confidence             : 1.000
  function-word zero-error    : 0.011% (n=17,707)
  coverage: 32% unambiguous, 28% ambiguous, 10% OOV

=== homograph-in-the-wild (native review — no auto ground truth) ===
  [говори] И незави́симо какъ́в законопрое́кт се гле́да ту́к, то́й ви́наги ка́зва: и́два господи́н Ива́нов и гово́ри едни́ не́ща по еди́н законопрое́кт.
  [говори] Господи́н Председа́тел, кога́то се гово́ри за избира́телни се́кции в чужби́на и в Репу́блика Ту́рция, това́ е сра́мен акт на полити́ческа репре́сия и ВЪЗРА́ЖДАНЕ е не́говият на́й-шумен и на́й-безсрамен идеоло́г.
  [говори] Кога́то се гово́ри от та́зи трибу́на, се гово́ри от и́мето поне́ на оне́зи, ко́ито са гласу́вали, а не про́сто от свое и́ме, за да се пу́снем по́сле в социа́лните мре́жи и да събере́м ня́колко ла́йкове.
  [говори] Ча́ст от вноси́телите, за ко́ито гово́ри господи́н Червенко́ндев, са вноси́тели на ма́лки коли́чества, ко́ито слу́жат за задоволя́ване на конкре́тните ну́жди на конкре́тни произво́дствени предприя́тия.
  [носи] Кога́то е оба́че на дру́ги държа́ви или́ но́си оръ́жие, тога́ва вече́ и́ма дру́га процеду́ра, ко́ято ни́е ви́наги сме спа́звали.
  [носи] И ко́й, господи́н Мини́стър в оста́вка, ще но́си отгово́рност за вси́чки те́зи не́ща, ко́ито сте сътвори́ли в то́зи зако́н?
  [носи] Но да не забра́вяме, че Цице́лков е вка́ран въ́тре от Гю́ров, ко́йто но́си отгово́рността за това́.
  [носи] Тя́ е водени́чният ка́мък на ши́ята на мини́стър-председателя, то́й да реши́ докога́ ще но́си водени́чните си ка́мъни.
  [прави] И ни́е го ви́ждаме все́ки ден, във все́ки е́дин се́ктор – тури́змът не прави́ изключе́ние.
  [прави] И знае́те ли защо́ се пра́ви това́?
  [прави] В Бълга́рия това́, ко́ето ще се слу́чва по до́говора, тря́бва да се пра́ви една́ мно́го я́сна ра́злика.
  [прави] А сега́шният им мини́стър пък да се пра́ви на тру́дно по́мнещ и да забра́ви, че всъ́щност ти́я сигна́ли са би́ли по вре́мето пак на не́говата па́ртия при Бо́йко Ра́шков.
  [учи] Уважа́еми да́ми и господа́, Рамаза́н Байрам ни учи́ да отва́ряме сърца́та си не са́мо към бли́зките си, но и към все́ки, ко́йто и́ма ну́жда от подкре́па.
  [учи] И не́говите бра́тя сега́ Европе́йският съю́з ще ги у́чи ка́к да си сме́нят пола́.
  [купи] Ако́ си ку́пи не́що с ка́рта, то му изли́за по́-скъпо, отко́лкото да си го ку́пи в бро́й.
  [купи] Спече́лил десе́тки меда́ли и ку́пи от състеза́ния в страна́та и чужби́на.
  [купи] Ако́ си ку́пи не́що с ка́рта, то му изли́за по́-скъпо, отко́лкото да си го ку́пи в бро́й.
  [чета] Чета́ в моти́вите, че Законопрое́ктът се нала́га заради́ установе́на поро́чна пра́ктика от страна́ на некоре́ктни опера́тори, ко́ито в действи́телност не извъ́ршват де́йности с отпа́дъци, а реа́лно извъ́ршват престъпле́ния.
  [чета] Със си́гурност това́, ко́ето Ви че́та, е логи́чно, ка́к да ка́жа, под ня́каква фо́рма да сме прово́дник на не́що, ко́ето тря́бва да изри́не та́зи па́смина.
  [само] Между́ дру́гото, в Слова́кия се гласу́ва са́мо и еди́нствено в държа́вата – не се гласу́ва извън Слова́кия.
  [само] Заслужа́ва си то́й да бъ́де подкрепе́н, но да не бъ́де затво́рен, специа́лно репре́сията да не бъ́де затво́рена са́мо в ра́мките на изпълни́телната вла́ст.
  [само] Ако́ бъ́де прие́т са́мо еди́н законопрое́кт, не се изго́твя общ докла́д, а се изго́твя докла́д са́мо на прие́тия законопрое́кт.
  [само] Защо́ дне́с не разгле́ждаме вси́чки внесе́ни законопрое́кти на те́ма управле́ние на отпа́дъците, а са́мо после́дният?

=== 20 random marked sentences (native read) ===
  И́скам да изразя́ едно́ ли́чно мне́ние.
  Благодаря́ Ви, госпо́жо Катинча́рова.
  Тога́ва отде́лът за закри́ла на дете́то, разпола́гайки с необходи́мите да́нни, и́ма пра́во да предприе́ме ме́рки за закри́ла.
  И вси́чкото това́ не́що го ви́ждаме в изка́званията на БСП.
  Това́ е една́ реа́лна по́мощ за земеде́лските производи́тели, ко́ято се по́лзва от почти́ вси́чки държа́ви – чле́нки в Европе́йския съю́з.
  „ДПС – Ново́ нача́ло“ от своя́ страна́ напра́виха кво́рума на Пра́вната коми́сия.
  Ка́зах го, свла́чището е акти́вно.
  Но така́ или́ и́наче, това́ повише́но потребле́ние ка́то че ли не се полу́чи, въ́преки си́лно завише́ните разхо́ди за заплати́ и за социа́лни ангажиме́нти.
  Наи́стина то е зна́к за уваже́ние към наро́дното предста́вителство.
  Ръководи́тел на МВР мо́же да бъ́де гра́жданско лице́, ко́ето не е рабо́тило в систе́мата на МВР и и́ма съ́щите кадрови́ правомо́щия, какви́то би и́мал и ръководи́тел на МВР, ко́йто е рабо́тил въ́тре в систе́мата.
  Тога́ва реше́нието ще ста́не: „За и́збор на секрета́р на Четири́десет и пъ́рвото Наро́дно събра́ние”.
  Така́ не се бо́ри кору́пция.
  Ако́ при пъ́рвото гласу́ване ни́кой от кандида́тите не полу́чи необходи́мото мнозинство́, се произве́жда вто́ро гласу́ване, в ко́ето уча́стват два́мата кандида́ти, полу́чили на́й-много гла́сове.
  При пъ́рвото гласу́ване да́дох дори́ две́ мину́ти – изча́ках извъ́н предви́деното вре́ме за гласу́ване.
  Уважа́еми господи́н Си́деров, в това́ Наро́дно събра́ние и във вси́чки наро́дни събра́ния, ко́ито изми́наха, се нагле́дахме на хора́, ко́ито от трибу́ната гово́рят едно́, а де фа́кто пра́вят дру́го.
  Ру́мен Раде́в е чове́кът, ко́йто и́маше информа́цията за те́зи де́вет годи́ни за вси́чко, ко́ето ста́ва в държа́вата.
  За смети́щето, ко́ето е в природни́я парк, не ка́сае се́ч.
  Това́ мо́же да бъ́де ча́стна компа́ния, ча́стно лице́, въ́ншен изпълни́тел.
  Но па́к повта́рям, пробле́мът с легити́мността в моме́нта е полити́чески, а не форма́лно юриди́чески.
  Уважа́ема госпо́жо Председа́тел, уважа́еми да́ми и господа́ наро́дни предста́вители!
 iff `y_{v*} ≠ 1`. `SER = errors /
  N_single`. Reported with its denominator `ser_n = N_single` (≈102,686 on
  our val split). A hard 0/1-per-word metric (contrast with `val_loss`):
  0.025% ≈ 1 misplaced stress per 4,000 words. This is the primary number to
  quote and to compare against the paper's auto/manual SER (0.119% / 0.37%).

- **`zero_err` — spurious-stress rate on unstressed words (the complementary
  metric).** The opposite failure mode: a word that should be entirely flat
  but received a stress. Evaluated on *fully-labeled zero-stress words*
  (function words / clitics labeled all-0, `zero_n` ≈66,273); a word is an
  error iff ANY of its vowels has `σ(z_v) ≥ 0.5`. `zero_err = errors /
  N_zero`. It is reported separately from SER because the two error types are
  asymmetric — a content word must place *exactly one* stress on the *right*
  vowel, a clitic must place *zero* stress on *any* vowel — and folding them
  into one number hides clitic regressions. This metric is what exposed the
  rechko-clitic bug (и→и́, му→му́ stressed in running text); driving it to
  ~0.000% is the objective confirmation of the function-word precedence fix
  (`mark_chitanka_text.load_function_words`).

A complete model needs all three low simultaneously: `val_loss` says the
probabilities are sharp and calibrated, `SER` says content words are stressed
on the right syllable, `zero_err` says function words stay flat. The metrics
are near-independent — a model can ace `SER` while failing `zero_err`
(over-eager stressing), the exact state the re-mark corrected. `best.pt` is
selected on lowest `SER`.

> Homograph accuracy (the battery, §5b) is a SEPARATE evaluation from these
> three — SER/zero_err/val_loss are measured on words with a KNOWN single
> stress, whereas homographs are masked during pretrain. The fine-tune
> (stage 6e) adds the speech-labeled homograph tokens; report battery
> pass-rate alongside SER, never merged into it.

### Out-of-domain evaluation on parliament stenograms (`eval_accentor_ood.py`)

The train/val split is all chitanka FICTION, so val SER cannot show
generalization. Parliament stenograms are formal POLITICAL speech — a
different register, vocabulary, and syntax the Accentor never trained on —
which makes them a true held-out generalization test. Two metrics on unseen
text:

1. **Automatic OOD SER (no human).** For every UNAMBIGUOUS lexicon word in the
   sampled parliament sentences, compare the model's argmax-stressed vowel to
   the dictionary's. Measures whether learned lexical stress transfers to a
   new domain/vocabulary. Robust to any sentence overlap — it scores dictionary
   words, not memorized sentences. Reported with denominator + function-word
   zero-error. Deliberately excludes homographs (no automatic ground truth
   without the audio).
2. **Homograph-in-the-wild (native review).** Sentences containing the
   homograph verbs, marked by the model, for a native to judge the contextual
   choice on natural text — more representative than the 25-item battery.

```bash
./venv/bin/python eval_accentor_ood.py --ckpt exp/accentor_ft3/best.pt \
    --transcripts data/bg_parliament/transcripts --n 4000 --show 25
```

**Result (ft3, 2026-07-07 — a strong generalization finding for the paper):**
OOD SER **0.014%** on 14,251 unambiguous parliament words (≈2 errors in 14K),
mean confidence 1.000, function-word zero-error 0.000% — essentially no
degradation from fiction to political speech. Homograph-in-the-wild: говори /
прави / чета / само / купи read correctly in context (чета→чета́ verb in "което
Ви чета́"; the failing battery sense че́та=band is genuinely rare in
parliament); носи / учи show the imperfective over-final error in the wild too
("да се носи́ вина" should be но́си), consistent with the battery. Use it as the
standard regression eval for every checkpoint, and re-run after the balance
experiment (OOD SER should hold ~0.01%; watch whether носи/учи wild-errors
clear).

---

## Stage 4 — Parliament crawl (runs unattended)

```bash
# start / RESUME (identical command — state/*.ok + manifest make it idempotent;
# '>>' appends the log). After any unclean shutdown: rm -rf data/bg_parliament/raw/*
nohup ./venv/bin/python prep_bg_parliament.py --out data/bg_parliament \
    --date-start 2020-01-01 --date-end 2026-12-31 --id-start 10425 --id-end 11150 \
    --reverse --sleep 1.0 --min-free-gb 200 --audio-format flac \
    >> data/crawl_2020_2026.log 2>&1 &

# progress: use the /crawl-status skill, or:
tail -f data/crawl_2020_2026.log

# post-pass (run every ~2-4 weeks): repair sessions crawled before their
# stenographic transcript was published (LOW c/s). Dry run first:
./venv/bin/python refresh_bg_parliament.py --out data/bg_parliament
./venv/bin/python refresh_bg_parliament.py --out data/bg_parliament --apply
```

Do **not** `--reset-high` sessions 11133/11135 — those are server-side partial
recordings; Phase-2 alignment discards the unmatched transcript tail anyway.

**Volume targets** (~100 MB/h at flac): milestone A ≈ 500 h raw (~50 GB) →
train the annotation TDNN (415 h of real audiobooks count toward the paper's
749 h AM recipe). Paper parity ≈ 3,900 h raw (~350 GB) → 2,695 aligned hours.

---

### Background: the two acoustic models (GMM → TDNN), in plain words

Kaldi's alignment uses two kinds of acoustic model, one after the other. Both
answer the same question — *"how likely is this 10 ms slice of audio to be
phone X (e.g. the stressed vowel a1)?"* — but with different machinery, and
we use them in sequence for a reason.

**GMM = Gaussian Mixture Model** (the name is the math; the idea is simple).
Each 10 ms slice becomes ~13 numbers (MFCCs, see the feature section). Plot
them and every instance of a phone lands in a *cloud* of points — not one
exact spot, because every "a" differs by speaker, pitch, mood. A **Gaussian**
is a bell-curve blob saying "sounds like this cluster here"; a **mixture** is
several blobs together (one phone has sub-varieties). So a GMM for `a1` is a
handful of blobs scoring how well a slice matches. It is 1980s technology, not
a neural net: CPU-trainable in hours, robust on modest data, but it models
each phone somewhat in isolation, so it is a bit crude. This is the
`mono → tri1 → tri2b → tri3b` ladder — the workhorse that does our forced
alignment and has manufactured the 1.22 M stress labels so far.

**TDNN = Time-Delay Neural Network** (the neural upgrade). "Time-delay" is the
point: instead of scoring each slice alone, it looks at a **window of
neighbouring slices together**, because a sound is defined partly by its
surroundings — the "a" in *мама* genuinely differs from the "a" in *врата*,
and a context-aware network captures that where a per-phone GMM cannot. It is
a real neural net: richer patterns, but needs more data and trains on the
**GPU** (hence the CUDA-13 rebuild). Crucially it is **not** trained from
scratch on raw audio — it learns from the alignments the GMM already produced
as its answer key.

**Why both, in order** — like focusing a lens in two passes: (1) the GMM is
cheap, CPU, robust, and gets the audio roughly lined up with the words,
yielding usable labels immediately; (2) the TDNN, trained on the GMM's
alignments, produces *sharper* alignments, hence *cleaner, more trustworthy*
stress labels. For us the payoff is precision: better alignment means the
aligner's а́-vs-а choice at each homograph is more reliable, so the labels
feeding the Accentor are better. That is exactly what milestone-A stage 6
buys — replacing the good-enough GMM labels with better TDNN labels before the
final Accentor training. (Forced alignment itself — the Viterbi search over
the pronunciation-variant FST — is the same operation under either model; the
TDNN only makes the acoustic scores it searches over more accurate.)

## Stage 5 — Kaldi alignment (pilot = audiobooks; parliament joins later)

One-time Kaldi install (CPU build; OpenBLAS `TARGET=HASWELL USE_LOCKING=1
USE_THREAD=0`, configure `--use-cuda=no --openblas-root=...`): see Pitfalls.

```bash
# stage 0: Kaldi data dirs + pronunciation dict.
#   - dict is VOCAB-LIMITED to transcript words (all stress variants kept).
#     Never use --full-lexicon on this machine (OOM — see Pitfalls).
#   - single-narrator corpora get pseudo-speakers (--spk-chunk 200);
#     parliament will use REAL speakers from turns/*.jsonl instead.
# stage 1: prepare_lang   stage 2: MFCC+CMVN   stage 3: GMM ladder
#   (mono -> tri1 -> tri2b LDA+MLLT -> tri3b SAT -> align_fmllr; overnight)
# stage 4: harvest pronunciation choices (get_prons.sh + battery split report)
KALDI_ROOT=~/kaldi bash kaldi_pilot.sh            # all stages
KALDI_ROOT=~/kaldi stage=3 bash kaldi_pilot.sh    # resume from stage N

# per-token stress labels -> JSONL for vetting + fine-tuning
./venv/bin/python harvest_stress_labels.py \
    --pilot exp/kaldi_pilot --ali exp/tri3b_ali \
    --out data/kaldi/stress_labels.jsonl
```

Health numbers from the completed pilot (63 h, 37,299 utts): final align prob
≈ −43.6, retry ≤ 2%, fail ≈ 1.3% (failed clips just contribute no labels).
Observed contextual splits: вълна́ 35/въ́лна 21, па́ра 24/пара́ 1,
гово́ри 125/говори́ 9, чета́ 9/че́та 5.

**Vetting protocol (native speaker):** from `stress_labels.jsonl`, listen to a
few tokens per variant (`audio_path` at offset `t0`) for the top split words;
judge whether the aligner's choice matches the narrator. GMM labels are noisy —
treat as provisional until the TDNN pass (stage 6).

### Full-bg400 alignment run (2026-07-12) — transcript OOV anatomy (paper material)

After the complete ~400 h bg400 ingest (195,701 manifest rows), stages 0–2
were re-run over the combined audiobook corpus: **203,863 utts (~450 h),
1,020 pseudo-speakers; dict vocab-limited to 184,345 pronunciations
(114,840/171,449 transcript types in lexicon; 3.17M out-of-transcript
pronunciations dropped)**. `tri3b_comb` stays compatible after the lang
rebuild (identical 34+sil phone set) — alignment proceeds with NO GMM
retrain (same-narrator data adds nothing a SAT model needs; the one retrain
that matters is the milestone-A TDNN seed).

**OOV analysis (measure this per corpus — the structure is the finding):**
56,609 transcript types (33% of types) but only **226,040 / 4,042,683 tokens
= 5.6%** map to `<unk>`. The token-level rate is what matters (OOV tokens
just yield no labels; the type inventory of ASR transcripts is junk-heavy).
The top-15 OOV tokens decompose into exactly two classes:

1. **Proper names of the narrated fiction** (елора, елейн, перин, ранд,
   седай, дънкан… — a Wheel-of-Time narrator), ~3–4% of tokens.
   Genuinely out-of-dictionary; unfixable from rechko; harmless as `<unk>`.
2. **Hyphenated по-/най- comparatives** (по-добре 1,257×, по-малко,
   по-късно, по-скоро…), 25,753 tokens = 11.4% of OOV = **0.64% of all
   tokens** (measured). A **lexicon gap, not a
   tokenizer bug**: `prep_kaldi_data.WORD_RE` keeps hyphenated tokens
   intact, but the candidates DB has NO hyphenated entries (по-добре → ∅
   while добре is present). The same words are masked-OOV in the text
   pipeline too — the Accentor's correct по-висо́ко on the battery is pure
   character-level generalization, not lexicon supervision.

**Paper points:** (a) label-manufacturing coverage is bounded by LEXICON
coverage of the transcript register, and the OOV structure is
register-diagnostic — fiction loses names, parliament will lose different
mass (verify at milestone A); (b) the comparative class is *recoverable by
rule* (по-X/най-X → prefix phones + X's candidate pronunciations) but
encodes a real linguistic decision first: comparatives arguably carry
DOUBLE stress (по́-добре́), while the whole pipeline assumes one stressed
vowel per word — the native speaker must fix the convention (prefix `o1`
vs `o0`, and what the harvested `marked` form should be) before the rule
is written.

**TODO (milestone A):** add the по-/най- prefix rule to
`prep_kaldi_data.write_dict` — `kaldi_milestone_a.sh` stage 0 rebuilds the
dict, so the fix lands there for free, where it matters more (по-добре is
very frequent in parliamentary speech). Decision for now: align as-is; the
lost 0.64% are ordinary regular formations, not homograph gold.

---

## Stage 5b — Parliament viability pilot (runs on the banked ~100 h NOW)

Proves the paper's §3 loop end-to-end on the real target corpus before the
crawl finishes. Requires the audiobook pilot (stage 5) completed.

```bash
KALDI_ROOT=~/kaldi bash kaldi_parliament_pilot.sh
# stage 0: long-utterance dir (1 utt = 1 session, spk = session) +
#          dict/lang rebuilt with audiobook+parliament vocab (same phones ->
#          the audiobook GMMs stay compatible)
# stage 1: MFCC on the long sessions, then segment_long_utterances.sh
#          (transcript-biased LM decode, seed = tri2b SI model; mismatched
#          regions — interjections, unverbalized numbers — are DISCARDED)
# stage 2: MFCC + align_fmllr of the segments against tri3b
#          (fMLLR per session = channel adaptation)
# stage 3: get_prons + harvest -> data/kaldi/stress_labels_parliament.jsonl
```

Viability gates: (a) segmentation retains ≥ ~50% of the 96 usable hours;
(b) homograph splits are context-plausible; (c) native vetting of ~20 tokens
per top split word passes. Segmentation runs at `seg_nj=8` (each job holds a
whole session transcript's biased LM — memory budgeting, Pitfall 1); expect
overnight for the decode. Numbers in transcripts are currently dropped, not
verbalized — the audio regions containing them are lost to segmentation
(acceptable for the pilot; verbalization matching is the paper's §3.1 upgrade,
planned for stage 6a).

**Round-1 result (2026-07-05): gates FAILED as expected for a cross-domain
seed** — 20% yield (23.8 h), and 10–35% per-token stress noise measured by the
built-in noise meter (minority counts on known single-stress words that the
recall-oriented lexicon over-generates: тези, или, това…; the audiobook pilot
reads ~4–7% on the same meter). Cause: audiobook-only GMMs vs parliament
acoustics. **Stages 4–6 are the in-domain bootstrap iteration**: retrain GMMs
on audiobooks + round-1 segments, re-segment (`data/parl_seg2`), re-align,
re-harvest (`stress_labels_parliament2.jsonl`), and re-read the noise meter.
This iteration IS the plan on any new acoustic domain; at milestone A the same
role is played by the annotation TDNN (stage 6b).

**Native vetting verdict (2026-07-05, 107 judged + 83 unsure across 16
words):** canonical-variant labels are ~95–100% accurate (гово́ри 5/5, чета́
5/5, страна́ 5/5, мо́же 6/6, съю́з 4/4…), and REAL contextual readings were
confirmed (говори́ 2/3, one true че́та; про́тив AND проти́в both 100% — genuine
variation). But the GMM aligner has a **systematic final-stress bias on
prosodically weak words, strong enough to invert corpus majorities**: може́
outnumbered мо́же 1125:517 yet vetted 0/5; всички́ 767:704 yet 0/6; съ́юз
~50:50 yet 0/7. Count-weighted accuracy is therefore per-word: това ~95%,
тези ~94%, чета ~79%, съюз ~49%, може ~31%. The 44% unsure rate is signal
too: these words genuinely reduce in fast speech.

**Fine-tune label policy (calibrated by the vetting):**
1. **Trust-canonical tier** — words whose canonical share is ≥ ~90% and the
   canonical variant vetted clean: keep canonical-variant tokens, DROP
   minority tokens (they are aligner noise): тези, това, бъде, време, има…
2. **Exclusion tier** — words with the final-stress-bias signature (canonical
   share < ~70% or vetted minority acc ≈ 0 with big minority): exclude from
   stress supervision entirely, treat like function words: може, всички,
   съюз, вече, още, един, една, или, колеги, между. They are not homograph
   targets; forced stress labels there are noise by construction.
3. **Contextual tier (the gold)** — curated homograph classes (battery +
   rechko verb pairs) where vetting confirmed both readings occur: keep BOTH
   variants as labels (говори/говори́, чета́/че́та, про́тив/проти́в), optionally
   after per-token vetting while volumes are small.
The paper's answer to tier-2 noise is the annotation TDNN (stage 6b) —
GMM piloting was for viability, and the bias it revealed becomes a
regression test for the TDNN (its counts for може/всички/съюз must flip to
the canonical majority, or it inherited the bias).

**Round-2 result (2026-07-05): VIABILITY CONFIRMED.** Yield 99.7 h / 126.8
usable = **79% keep-rate** (gate a passed); 210,802 ambiguous-word tokens over
10,623 types. Noise meter on content words: **0.5–6.4%** (зако́н 99.5%,
мини́стър unanimous, вре́ме 1.6%, бъ́де 2.2%, те́зи 6.4%) — audiobook level.
Residual noise concentrates in **prosodically weak frequent words**
(или 21%, вече 33%, между 20%, има 11%, and suspicious majorities on
всички/може/още) — these destress/reduce in running speech, so a forced
stress choice is partly meaningless there. **Exclude them from fine-tune
labels** (function-word list + per-type noise stats); they are not homograph
targets. Battery splits: гово́ри 125 / говори́ 8 (imperative rare in
parliament — plausible), чета́ 10 / че́та 4. Vetting sheet:
`vet_stress_labels.py` → listen per token via ffplay commands (fix note:
harvester now maps segment-relative times into the source recording).

---

## Stage 6 — AFTER THE CRAWL: scale-up to parliament (the paper's §3)

Run in this order once milestone A (~500 h raw) is reached. 6a–6c are
**implemented as `kaldi_milestone_a.sh`** (stages 0–8: volume gate →
full-crawl segmentation → the ONE combined GMM retrain (tri4b) →
clean_and_segment → annotation TDNN (nnet3 CE, deliberately non-chain so
get_prons/harvest timestamps stay at 10 ms) → F5 bias acceptance test →
harvest). Prereq for its stage 6+: a CUDA Kaldi rebuild (current build is
CPU-only; CUDA 12.x toolkit needed — see the script header). Do NOT retrain
GMMs incrementally before this: bg400 is the same narrator tri3b_comb knows
(alignment doesn't need it) and round-2 parliament noise is already at
audiobook level; the one retrain that matters is the TDNN seed.

**6a. Segmentation.** Parliament sessions are hours long; Kaldi needs
utterances. Use the paper's method via `steps/cleanup/segment_long_utterances.sh`
against the tri3b model: it bootstrap-decodes each session, matches the
stenographic transcript (`transcripts/*.txt`), and emits aligned segments.
Speakers come from `turns/*.jsonl` (real speaker names per turn) — no
pseudo-speakers here. Add the segmented set to `prep_kaldi_data.py --set`.

**6b. Annotation AM (TDNN).** Retrain on audiobooks + parliament (~750 h
combined, ≈ the paper's 749 h) with the same stressed/unstressed phone set:
GMM ladder rerun on the combined data → `steps/cleanup/clean_and_segment_data.sh`
pass → TDNN (nnet3 chain, e.g. the WSJ/librispeech tdnn recipe) on the cleaned
alignment. This is the model whose stress judgments the paper trusts
(auto-SER 0.119%).

**6c. Per-sentence stress decoding.** For every training sentence, build a
tiny decoding graph that encodes ALL stress candidates of every word
(the alignment lexicon already carries them) and decode with the TDNN —
`harvest_stress_labels.py` then converts choices to labels exactly as in the
pilot. This yields the BG-PARLAMA-style corpus: parliament (target 22M tokens)
+ audiobooks (48M in the paper) with contextual stress.

**6d. Dictionary trick (paper §3.4).** Append the 551K unambiguous dictionary
types as single-word training rows (we already do the sentence-scale analog in
`mark_chitanka_text.py`; add the explicit single-word rows at fine-tune time).

**6e. Accentor fine-tune.** Continue training `exp/accentor_text/best.pt` on
the speech-labeled rows (homographs now supervised, not masked) + re-marked
chitanka + dictionary rows. Gate: battery must go from 11/25 toward ~25/25;
hard-word set must stay perfect. Vet the battery output by ear before deploying.
First executed 2026-07-05 on the pilot labels (works the same at full scale):

```bash
# 1. re-mark chitanka (и́/му́ function-word precedence fix)  ~8 min
./venv/bin/python mark_chitanka_text.py \
    --corpus data/chitanka --sqlite data/bg_lexicon/bg_stress_candidates.sqlite \
    --out data/chitanka/marked.jsonl --max-tokens 80000000
# 2. retrain the text pretrain from scratch                  ~4.6 h GPU
./venv/bin/python train_accentor.py \
    --data data/chitanka/marked.jsonl --out exp/accentor_text2
# 3. speech rows under the three-tier policy + 5:1 chitanka mix
./venv/bin/python build_accentor_finetune.py \
    --labels data/kaldi/stress_labels.jsonl \
    --labels data/kaldi/stress_labels_parliament2.jsonl \
    --mix-chitanka data/chitanka/marked.jsonl --mix-ratio 5 \
    --out data/accentor_finetune.jsonl
# 4. fine-tune: weights from the pretrain, fresh low-LR schedule  ~30-60 min
./venv/bin/python train_accentor.py \
    --data data/accentor_finetune.jsonl --init exp/accentor_text2/best.pt \
    --out exp/accentor_ft --steps 10000 --lr 1e-4 --warmup 500 --eval-every 1000
# 5. gates: battery up from 11/25, hard words still perfect
./venv/bin/python eval_accentor.py --ckpt exp/accentor_ft/best.pt \
    --battery config/bg/homograph_battery_sentences.json
```

**What data trains what (state this precisely in the paper).** The pretrain
and the fine-tune differ fundamentally in the *source of their labels* — that
difference is the reproduction's central claim:

| | PRETRAIN (stage 3) | FINE-TUNE (this stage) |
|---|---|---|
| File | `data/chitanka/marked.jsonl` | `data/accentor_finetune.jsonl` |
| Size | ~7M sentences, 80M tokens | 458K rows (76K speech + 382K chitanka mix) |
| Label source | **dictionary** (rechko lookup) | **human speech** (Kaldi alignment, model #1) |
| Homographs | MASKED (excluded from loss) | **SUPERVISED** with real contextual readings |
| Teaches | stress on *unambiguous* words | stress on *ambiguous* words, in context |

The fine-tune set (`build_accentor_finetune.py`) is:
- **~76K speech rows / ~221K stress-labeled tokens** — the payload. Each is a
  real human-spoken sentence (audiobook narrators + parliament), where the
  per-word stress was chosen by the Kaldi aligner FROM THE AUDIO, harvested
  back to text. This is the one thing the dictionary cannot supply:
  which reading of a homograph is correct in *this* sentence.
- **~382K chitanka rows (the 5:1 mix)** — a random sample of the re-marked
  pretrain corpus folded back in as a RETENTION ANCHOR, so fine-tuning on the
  narrow speech set does not make the model forget the broad lexicon it
  already mastered. Not new information — regularization against drift.

Which speech tokens become labels is governed by the three-tier policy
(calibrated by native vetting, §5b, encoded in `build_accentor_finetune.py`):
tier-3 curated homographs keep BOTH aligner readings (the contextual gold);
tier-1 ordinary words are kept only if the token matches the canonical stress
AND the word is ≥90% canonical corpus-wide (aligner-noise gate); tier-2
prosodically-weak words with the GMM final-stress bias (може, всички, съюз…)
are excluded entirely. Observed tier-3 yield: гово́ри 250, чета́ 19 / че́та 9,
вълна́ 35 / въ́лна 22, па́ра 24 / пара́ 1 — both readings of each, from audio.

One-line summary for the paper: *the pretrain learns general stress from the
dictionary with homographs masked; the fine-tune adds the one thing the
dictionary cannot provide — contextual homograph disambiguation from aligned
human speech — plus a chitanka anchor to retain the general lexicon.*

**Fine-tune #1 result (2026-07-05) — the minority-reading coverage wall
(paper-critical finding).** Battery went 11/25 → 12/25; general SER held at
0.023% (zero forgetting), every majority reading learned at 1.00 confidence.
But EVERY battery failure is the *minority* reading of its class, because the
training data barely contains them — the counts are decisive:

| word | majority (learned ✓) | minority (failed ✗) |
|---|---|---|
| говори | гово́ри 250 | говори́ 17 |
| само | са́мо 2610 | само́ 36 |
| прави | пра́ви 461 | прави́ 13 |
| носи | но́си 97 | носи́ 5 |
| ходи / учи | хо́ди 21 / у́чи 10 | ходи́ 2 / учи́ 2 |
| моли | мо́ли 4 | моли́ **0** |
| коса | коса́ 94 | ко́са **0** |

**Root cause is corpus REGISTER, not the model or method.** Parliamentary and
audiobook-narration speech is overwhelmingly declarative 3rd-person, so the
imperative reading of the -и verb class (говори́!, носи́!, прави́!) structurally
barely occurs. The aligner captured every one it heard; the corpus just holds
few. The one genuinely balanced class behaves perfectly — чета́ 19 : че́та 9,
where the model correctly shows honest uncertainty (0.51). **Publishable
claim: homograph-disambiguation accuracy is bounded by minority-reading
coverage, and corpus register systematically starves specific homograph
classes** (here the entire imperative/3sg alternation). Reporting rule: quote
battery accuracy split by majority vs minority reading, never merged.

**Fix ladder (in leverage order):**
1. Tier-3 now includes the -и verb classes (само, моли, носи, прави, купи,
   учи, ходи) so their few real minority tokens stop being gated out —
   necessary but not sufficient (can't learn моли́ from 0, ходи́ from 2).

**Fine-tune #2 — the floor measurement (2026-07-05, decisive for the spec).**
Rebuilt with the tier-3 verb expansion, re-fine-tuned (`exp/accentor_ft2`,
SER 0.020%, no forgetting). Battery stayed 12/25 but two classes moved in
opposite, diagnostic directions:
- **чета 1/2 → 2/2 (proof-of-mechanism):** with ~balanced tokens (чета́ 19 :
  че́та 9) fully supervised, чета́ went from a 0.51 coin-flip to a confident
  0.99 PASS on a HELD-OUT sentence. Sufficient balanced data → it generalizes.
- **купи 1/2 → 0/2 (cautionary):** admitted to tier-3 with sparse, subtle
  tokens (ку́пи 21 : купи́ 9; aorist vs imperative are segmentally identical),
  the model learned both readings BACKWARDS. Sparse/noisy minority data can
  DEGRADE a class that was accidentally right by prior — "a little noisy data
  is worse than none."
- Imperatives with 0–5 tokens unchanged (still fail); semantic homographs
  (вълна/пара/коса/само) unchanged (not morphological — need speech volume).

**Spec this hands the miner:** the wall IS data volume (чета flipped at ~20+
balanced tokens); mining is REQUIRED, not optional; and it must deliver **~20+
distinct contexts per reading** — a handful can hurt (купи). The чета
threshold (~20/reading) is the target count. Corollary: do NOT admit a class
to tier-3 until it can be populated to that floor.

**Fine-tunes #3–#4 — both-readings mining + the absolute-count wall
(2026-07-07):** #3 (both-readings mine, unbalanced) flipped **говори (141 imp)
and купи (49, perfective) to 2/2** — first imperative flips — and made all 7
verb-homograph 3sg readings solid; battery 12/25 (semantic homographs чета/пара
regressed, offsetting). #4 rebalanced root dominance 40:1→6:1
(`--balance-ratio 4 --balance-floor 300`): battery **13/25**, говори/купи held,
OOD SER held 0.021% (the 2000→300 3sg cut did NOT hurt) — **but прави did NOT
flip at 6:1**, nor носи/учи/ходи/моли. **Decisive finding: the imperfective-
imperative wall is ABSOLUTE minority-reading COUNT, not class ratio** — говори
crossed at 141 distinct contexts, прави failed at 50 even balanced. Reweighting
≠ new signal. Lever from here = more *distinct* imperative data (milestone-A
parliament speech), not tuning. Perfective imperatives (купи/скочи) and
high-count imperfectives (говори) are solved; low-count imperfectives are
data-bound. See §3 OOD eval for the 0.021% generalization number.

**Fine-tune #5 — the full-400h run (2026-07-12): battery 13→17/25, OOD SER
0.005% (best ever), and three findings sharpened.** Data: the full-bg400
harvest (`stress_labels_ab400.jsonl`, 1.22M tokens — supersedes the 63 h
`stress_labels.jsonl`, do NOT stack both) + parliament round-2 + mined
contexts + 5:1 chitanka anchor = 1.44M rows (239,788 speech rows / 947,697
speech-labeled tokens). Pre-flight vetting (36 variants, 164 judgments):
па́ра/пара́ and са́мо/само́ 100% clean → admitted; every tier-2 exclusion showed
the 0%-minority signature (може́ 0/5, съ́юз 0/7, всички́ 0/6, вече́/още́/има́ 0%)
while canonicals vetted ~100% — the strongest F5 evidence table yet; **като
vetted като́ 10/10, ка́то 0/9** — the harvest majority was RIGHT and the
Phase-A recollection ("narrator says ка́то") wrong: *vet, don't remember*.
Training: 10K steps, val SER 0.025% (n=84,425), zero_err 0.000%, SER still
descending at 10K (~1.8 epochs of the 3× corpus).

Result analysis (each line updates a finding):
- **прави́ 2/2 — the F4 wall broke between 50 and ~99 contexts:** ~46 vetted
  speech + ~53 mined ≈ 99 total; говори crossed at 141, прави failed at 50,
  passes at ~99. The absolute-count threshold is in **[50, 100]**.
- **ходи́ 2/2 with only 3 speech tokens** — the -и imperative pattern has
  begun to TRANSFER across verbs once enough classes (говори, прави, купи)
  are solid: per-class floors drop as the paradigm fills in.
- **чета INVERTED (F3 confirmed live, predicted by vetting):** че́та tokens
  vetted 33% (1/2/1) pre-training; after training че́та passes but the
  previously-solid чета́ verb FAILS, and wild parliament text shows the model
  wavering (че́та/чета́ in one sentence). Label precision gates minority
  admission — the vet verdict predicted the inversion before the GPU did.
  Fix path: per-token vet the ~30 че́та tokens, drop the bad, retrain.
- **Vetted-clean ≠ flips: пара́ (24 tokens) and само́ (101) did not flip.**
  пара́ sits AT the ~20 floor, not above it; само́ is drowned 100:1 by са́мо
  10,759 — at extreme imbalance, ratio still matters even though F4 says
  count dominates in the 6:1 regime. New passes elsewhere: па́ра (steam),
  че́та (band), Христо́, Ку́пих aorist.
- **Production regressions (pin via overrides until next round):**
  промени́те for проме́ните (native-confirmed) in the TrainGen-register hard
  words ("Запишете промените") — pinned via `bg_stress_add.py промЕните`
  (kills the rare промени́те imperative reading; right trade for narration).
  Also бо́я for боя́ (paint) — but боя is a true homograph (бо́я = the fight,
  def. of бой), so pinning it is a native call, not automatic.
- OOD SER 0.005% on 18,565 unambiguous parliament words, function-word
  zero-error 0.000% — fiction→politics transfer improved 4× alongside the
  battery gains (no tension between general SER and homograph learning).

**Fine-tune #5b — the vetting veto + boundary oscillation (2026-07-13).**
Full че́та vetting (30/variant) measured the harvest's че́та tokens at **19%
precision** (5 ok / 21 bad / 4 unsure) vs чета́ 100% (33/33) — the inversion's
smoking gun. Fix: `build_accentor_finetune.py --vet-results` (new) — every
harvest token the native judged bad/unsure is dropped BEFORE tier logic
(169 tokens removed; keyed on utt+word+marked+t0). The vetting CLI is now
literally part of the training-data path: humans calibrate, machines scale.
Retrained 20K steps (~1.8→3.6 epochs): **val SER 0.020% (best fine-tune
ever), OOD SER 0.005% (step 11K) / 0.000% (step 20K)** — and the wild-text
чета́ inversion is FIXED at the best.pt checkpoint.

**Boundary-oscillation finding (paper material):** battery classes with
20–100 training contexts sit on the decision boundary and FLIP between
adjacent checkpoints of the same run — battery 17 (ft5 @10K) → 15 (ft5b
@11K) → 16 (ft5b @20K), with пара changing direction twice, вълна swapping
which 2-of-3 pass, говори́/чета́ landing at 0.49–0.55 confidence. The stable
core (all 3sg readings, прави́/ходи́/купи́ imperatives, коса́, коли́, за́мък)
never moves. **Checkpoint selection is blind to this: best.pt selects on
general val-SER, which does not include homographs.** Checkpoint choice for
serving must therefore weigh WILD-text homograph reads, not the battery
alone: step-20K scored +1 battery (пара 2/2, first ever) but regressed wild
чета́ (че́та в медиите) and wild купи́ ("да си купи́ ваксина" — да-subjunctive
demands ку́пи); step-11K is wild-perfect. **Serving choice: ft5b best.pt
(step 11K).** Fix path for the oscillation: (a) add battery+wild pass-rate
to checkpoint selection; (b) more distinct contexts per boundary class
(milestone A) — reweighting/threshold tricks do not add signal.
промените postscript: three checkpoints produced three different markings
(про́мените / промени́те / про́мените; native truth = проме́ните) — a textbook
boundary-class signature and the concrete case for the human-overrides veto
layer in serving (`bg_stress_add.py промЕните` pinned it; mark_tts_txts
applies overrides too, so the TTS trains on the pinned form).

2. **Imperative text-mining (6e-bis, the real fix)** — see below.
3. Milestone A: the full 415 h of real audiobooks (narrated *dialogue* = spoken
   imperatives, unlike parliament) + more parliament surface minority tokens
   naturally through the same alignment loop.
4. Class-balanced sampling once (2)/(3) provide enough distinct minority
   contexts that up-weighting generalizes instead of memorizing 2–17 rows.

**6e-bis. Imperative text-mining (planned — supplements sparse speech labels
with rule-labeled text).** Key linguistic fact: the imperative of the -и verb
class is a REGULAR morphophonological alternation — the singular imperative is
**always final-stressed** (говори́, носи́, прави́, учи́, ходи́, моли́, купи́),
versus the root-stressed 3sg present (гово́ри, но́си, пра́ви…). Because the
stress is rule-predictable once the imperative MOOD is identified, we do not
need audio for this class — we need to find imperative *contexts* in text,
which literary dialogue (chitanka) has in abundance and parliament lacks.

BUILT (`mine_homographs.py` + `config/bg/homograph_verbs.json`). **Mines BOTH
readings** of each verb homograph, not just the imperative — a context-aware
model learns the boundary only from correct labels of both, so rather than
purify the hard imperative side we label each reading via its own
high-precision cue and SKIP the ambiguous middle (a wrong label inverts a
class). Marking is decoupled from mining: correctness from the curated
lexicon, coverage from the text.

- **3SG (root, high-yield/clean):** subject/subordinator before the verb
  (той/тя/то/да/че/ще/който…) or a proclitic clitic (го носи, им говори).
  The `да`+verb subjunctive is the dominant, near-100% cue. Works in narration.
- **IMPERATIVE (final, sparse):** particle before (хайде/недей/моля…); for
  PERFECTIVE verbs (купи, скочи) also clause-initial in a short '!' dialogue
  clause (clean immediate command).
- **SKIP (ambiguous):** `не`/`я` before (negation / particle-pronoun clash);
  a bare clause-initial '!' on an IMPERFECTIVE verb — the indignant/habitual
  3sg exclamation ("Хо́ди по жени!", "Но́си му се славата!") is surface- AND
  meaning-identical to the durative imperative, unlabelable by rule.

200-book yields confirm the honest distribution: 3sg abundant (говори 2002,
прави 2001, носи 703…), imperative rare (говори 6, купи 3, most 0–2) —
imperfective imperatives are genuinely scarce. Precision ~96% (spot-checked).
`--show N` prints labeled rows for native review. Verbs (8, DB-verified marks):
купи/скочи (perfective) + носи/ходи/говори/прави/учи/моли (imperfective).
Earlier single-reading `mine_imperatives.py` is superseded/removed.
- **Label:** only the target token is supervised (marked + unmasked); every
  other word masked — a pure one-token teaching signal in `marked.jsonl`.
- **Validate:** `--show N` prints random labeled rows; native spot-check
  precision before training.
- **Use:** `build_accentor_finetune.py --extra-marked data/chitanka/homograph_contexts.jsonl`
  appends them to the fine-tune. The abundant, clean 3sg examples teach the
  root reading; the sparse imperative examples + the model's context-awareness
  do the rest. Aspect prediction stands: perfective (купи/скочи) mine cleaner
  imperatives than the durative imperfectives.

Prior single-reading design (`mine_imperatives.py`, particle+clause-initial
cues, ~97% precision but imperative-only, some indignant-3sg leakage) is
SUPERSEDED — the both-readings miner subsumes it and removes the false-positive
class by never labeling the ambiguous exclamatory 3sg.

Caveat for the paper: this is a targeted, rule-assisted augmentation for a
*morphologically regular* class. It does NOT generalize to lexically arbitrary
homographs (въ́лна/вълна́, па́ра/пара́, за́мък/замъ́к) whose disambiguation is
semantic, not morphological — those still require human-speech labels at
volume (milestone A). State both explicitly.

**6f. Deployment.** Swap the Accentor in front of OmniVoice: replace the
rechko lookup in `serve.py`/`bg_accent.py` with `accentor_model.mark_text()`
(keep the overrides JSON as a final veto). Then re-mark the OmniVoice finetune
data with the Accentor and re-finetune OmniVoice-BG (BG_FINETUNE_PLAN.md) so
the TTS was trained on the same marking it will see at inference.

---

## Stage 7 — Serving & TrainGen

```bash
# launch (defaults: models/omnivoice-bg, rechko lexicon + overrides, :8002)
bash serve_bg.sh
# variants:
OMNIVOICE_BG_MODEL=models/omnivoice-bg-v2 bash serve_bg.sh
OMNIVOICE_BG_REF_AUDIO=data/bg_ref/narrator.wav \
  OMNIVOICE_BG_REF_TEXT="$(cat data/bg_ref/narrator.txt)" bash serve_bg.sh

# smoke test
curl http://localhost:8002/health
# {"status":"ok","sampling_rate":24000,"fixed_voice":...,"bg_accent":true}
curl -s http://localhost:8002/synth -H 'Content-Type: application/json' \
    -d '{"text":"Натиснете бутона и въведете паролата."}' --output test.wav
```

`POST /synth` body: `text` (required) + optional `instruct`, `language`,
`engine` (`"omnivoice"` | `"bgtts"` when launched with `--bgtts-dir`), decode
params. Text is stress-marked server-side; the client sends plain Bulgarian.

**TrainGen** (`~/WORK/TrainGen`, `.env` on the TrainGen machine):

```env
TRAINGEN_TTS=omnivoice_remote
TRAINGEN_OMNIVOICE_URL=http://192.168.78.101:8002
TRAINGEN_VOICE_DIR=assets/voices
TRAINGEN_OMNIVOICE_AUTOVOICE=0
```

`traingen/tts.py::RemoteOmniVoiceBackend` POSTs plain text and writes the WAV.
Voice precedence: a server-side `--ref-audio` wins over everything; otherwise
per-voice clips in `TRAINGEN_VOICE_DIR`; otherwise the fine-tuned model's
default narrator. Pick ONE approach (server-fixed vs client per-voice).

---

## Stage 8 — Post-finetune polish: RL / preference optimization (explored 2026-07-05, planned)

RL-style methods apply to the **TTS only**. For the Accentor, supervised
fine-tuning is strictly better — its ground truth (dictionary + aligned
speech) is cheap and abundant; RL earns its complexity only where the
objective can't be expressed as labels.

For OmniVoice-BG the practical form is **not RLHF but RLAIF / preference
optimization with automatic rewards**, because this project already owns the
verifiers a reward model would have to learn:

| Reward | Verifier (already built) | Fixes |
|---|---|---|
| Stress obedience | the Kaldi stress-aware aligner: does the audio realize the U+0301 marks it was given? | the core TrainGen complaint |
| Intelligibility | Whisper CER vs input text | dropped/repeated words on long sentences |
| Speaker similarity | `omnivoice/eval/speaker_similarity` | cloning drift over long batches |
| Naturalness | UTMOS (`omnivoice/eval/mos`) | weak signal, use as tiebreak |

**Method ladder** (16 GB GPU reality — costs are dominated by candidate
generation, ~32 denoise steps per sample):

- **Rung 0 — best-of-N at inference** (zero training): `serve.py` generates
  2–3 candidates, scores with the battery, returns the winner. TrainGen is
  offline batch narration, so 2–3× latency is painless. Ship first.
- **Rung 1 — best-of-N distillation (RAFT)**: generate N candidates per
  sentence offline (5K sentences × 3 ≈ a couple of overnight runs), keep
  reward-winners, fine-tune with the ORDINARY training objective. Reuses the
  existing trainer; memory = the finetune we already ran. **Recommended
  target.**
- **Rung 2 — DPO**: same pairs, preference loss vs a frozen reference.
  Caveat: OmniVoice is a mask-prediction diffusion LM — exact sequence
  log-probs are intractable, so a masked-CE/ELBO proxy is needed
  (Diffusion-DPO style). Research-flavored; only if Rung 1 saturates.
- **Rung 3 — online PPO/GRPO: NOT on this hardware.** Fresh rollouts inside
  the training loop (denoise + codec decode + ASR scoring per sample) cost
  days-to-weeks per useful update budget on a 16 GB card.

**Vetting vs RLHF** (the distinction that shapes the design): the vetting CLI
is human feedback on *labels* — data curation for supervised learning; ~100
judgments suffice because they *calibrate* a high-precision automatic labeler
rather than carry the signal. RLHF is human feedback on *model outputs*
pushing the policy directly, needing thousands of judgments. Principle:
**humans calibrate, machines scale** — spend native-speaker time auditing
verifiers, not being one. Exception: *accent* is the one quality the
automatic battery captures worst — if human preference data is ever
collected, spend it all there (a `vet_stress_cli`-style A/B tool over
candidate pairs, used for best-of-N filtering, no reward-model training).

**Sequencing discipline:** RL polishes residuals. Run it only AFTER the
supervised loop lands (Accentor speech-label fine-tune → re-marked OmniVoice
finetune) — polishing a base that is about to be replaced wastes the
GPU-weeks that candidate generation costs. Rung 0 is the exception: it is
base-independent and can ship anytime.

### Stage 8 validated — best-of-8 oracle experiment (2026-07-13)

The `bon_generate.py` → `bon_judge.sh` → `bon_score.py` harness ran the
oracle test on the 28 border sentences with `omnivoice_bg_full/6500`
(413 h + ft5b marks; serve-time decoding, seed the only knob —
`position_temperature=5.0` randomizes unmasking order):

- **single-sample 12/27 → judge-picked 19/27 = oracle 19/27** (sentences
  with all judgeable targets correct). The Kaldi stress-aligner judge
  captured **100 % of the selection headroom** — Rung 0/1 are validated.
- Target level: 73/82 targets correct in ≥1 of 8 candidates.
- The 8 unrecoverable sentences decompose into distinct failure classes.
  True prior dominance (mark right, TTS refuses all 8): прави́ & ходи́
  (final-stress imperatives), сгреши́, пеша́, въ́лна. But the full table
  also exposed **Accentor errors the TTS unanimously corrected**: пара́
  marked for *steam*, чета́ for the *noun*, у́чи for the *imperative*,
  Си́нът for синъ́т, plus tier-2 weak words (още́, ако́, между́) marked at
  inference — i.e. *unanimous disobedience audits the front-end*, and the
  judge's 19/27 is an obedience score, not a correctness score. By-ear
  review additionally found stress errors on single-variant words the
  judge cannot see. Full decomposition, fixes, and the milestone-A
  execution plan: **`BG_FINAL_EFFORT_PLAN.md`**; distilled lessons:
  **`BG_LESSONS.md`**; artifact map: **`BG_ARTIFACTS_INDEX.md`**.

Context that triggered this: the 63 h → 413 h acoustic scale-up
(`omnivoice_bg_full`, eval loss 3.88 → 3.43) barely moved mark obedience —
acoustic volume saturated (F12), pointing the residual at sampling/prior
rather than data.

Judge design update (2026-07-14): the reward table above is now a
two-stage **composite judge** — Whisper round-trip CER as a GATE
(intelligibility; forced alignment force-fits wrong-word audio), Kaldi
stress score as the RANKER (stress; ASR is blind to it — BG orthography
doesn't write stress). Full design + the ASR-circularity trap:
`BG_FINAL_EFFORT_PLAN.md` §P4.

---

## Pitfalls & lessons (hard-won — read before running anything heavy)

1. **OOM masquerading as a reboot (2026-07-04).** `compile-train-graphs` with
   the FULL 3.35M-pron lexicon holds ~2.6 GB × nj=24 ≈ 62 GB → the kernel
   killed the desktop session twice; looked exactly like a power-surge reboot
   (WM freeze from swap thrash, then login screen). Fix is permanent
   (vocab-limited dict); the lesson generalizes: **budget jobs × per-job RSS
   against 60 GB before any `nj=$(nproc)` fan-out**, and check
   `journalctl -k | grep -i oom` before believing hardware explanations.
2. **Synthetic audio cannot label homographs.** BgTTS picks the frequency-prior
   reading regardless of context, consistently across seeds. Human speech only.
3. **rechko's clitic entries are wrong for running text** (и́, му́). The curated
   function-word list must outrank the lexicon (fixed in
   `mark_chitanka_text.py::load_function_words`; re-mark pending).
4. **Kaldi build on this machine:** CUDA 13.2 unsupported → `--use-cuda=no`;
   OpenBLAS needs `make TARGET=HASWELL USE_LOCKING=1 USE_THREAD=0`; a failed
   `configure` leaves a truncated `kaldi.mk` (~29 lines; a good one is ~92) —
   rerun configure before `make`. Diagnostic scripts want a bare `python`
   (`kaldi_pilot.sh` ships a `shims/python -> python3` symlink).
5. **Crawl resumability contract:** finished sessions = `state/*.ok`;
   `raw/` is transient (delete freely when nothing is running); the manifest's
   last line can be torn by a hard stop — `refresh_bg_parliament.py` doubles
   as a manifest integrity check.
6. **rechko has multi-word and duplicate entries** — `prep_kaldi_data.py`
   drops them (Kaldi would parse the second word as a phone).
7. **Single-narrator corpora need pseudo-speakers** (Kaldi parallelizes by
   speaker; SAT wants per-speaker transforms on minutes, not 200 h). Parliament
   uses real speakers from `turns/`.
8. **`ffplay -ss` HANGS on the crawl's session FLACs** (ffmpeg writes FLAC
   without a seektable; ffplay's seek path never recovers, `ffmpeg -ss` input
   seek is instant AND sample-accurate — verified byte-identical vs
   decode-from-start). Never seek with ffplay on these files: extract the
   clip with ffmpeg first, then play the small wav (`vet_stress_cli.play()`).
   Also wrap every player call in a hard timeout — a wedged audio client
   otherwise freezes the calling app.
   The subsequent "audio doesn't match text" report was NOT a coordinate bug:
   Whisper-transcribing a labeled window confirmed the harvest's
   segment→recording time mapping end-to-end (it even read the segment's
   `<unk>` back as the dropped number "172-ма"). Perceived mismatch = tight
   clip window playing neighbouring words (UX; default pad now 1.0 s + [w]
   wide replay) plus a REAL minor label-noise class: tokens that are
   fragments of a longer spoken word near OOV/alignment boundaries (има vs
   spoken имат) — filter tokens adjacent to alignment errors at fine-tune
   label selection (stage 6a).
10. **`prons.*.gz` are only valid against the lang they were written with.**
    `harvest_stress_labels.py` maps integer word/phone IDs via
    `data/lang/{words,phones}.txt`; the parliament pilot's stage 0 does
    `rm -rf data/lang` and rebuilds it with combined vocab, shifting word IDs.
    Harvesting the OLD audiobook prons against the NEW lang gave 74%
    phone-map mismatch and silently mislabeled the survivors (2026-07-05).
    Fix: re-run `steps/get_prons.sh <data> <current-lang> <ali-dir>` before
    harvesting any alignment made before the last `prepare_lang`. The
    harvester now warns when the mismatch rate exceeds 10%.
9. **The desktop session can be silently rebuilt** (e.g. after an OOM kill —
   Pitfall 1), and the default audio sink may move to another device
   (4 sinks on this box; it landed on the Logitech headset). If audio "plays"
   but is inaudible: `pactl get-default-sink` / `pactl set-default-sink …`.
   Even with the default sink correct, **ffplay's SDL audio layer routed to a
   different (inaudible) device** while `paplay` was audible — programs that
   must be heard should play through `paplay` (PipeWire default sink,
   directly); `vet_stress_cli.py` does since 2026-07-05.
11. **Neural (nnet3) alignment: try `--use-gpu true` next time (milestone-A
    lesson, 2026-07-16).** For the *GMM* aligner, CPU with high `nj` wins —
    acoustic scoring is cheap and the FST/Viterbi search dominates. For a
    *neural* aligner the opposite holds: the **12.3M-param TDNN forward pass
    is the bottleneck**, and on CPU it is heavy — milestone-A stage 7 aligned
    the 750 h cleaned set with `steps/nnet3/align.sh --use-gpu false` (Kaldi
    default, 24 CPU jobs) and it ran ~2.5–3 h. `nnet3-align-compiled` also
    logs nothing per-utterance and writes `ali.*.gz` in bursts, so progress
    is invisible mid-run (watch for the files jumping to full size, not
    incremental growth — it is NOT stalled if CPU-time keeps accruing at
    ~95%). Next time (milestone B, `harvest_new_sessions.sh`) try
    `--use-gpu true`: fewer jobs, but each runs the forward on the GPU, which
    may beat 24 CPU jobs since the forward dominates. The GPU must be free
    (evict llama.cpp/BgGPT first).

## File map (added by this project)

| File | Role |
|---|---|
| `build_bg_lexicon.py` / `build_stress_candidates.py` / `build_lexicon_phones.py` | Stage 1 lexicon chain |
| `extend_lexicon_coverage.py`, `lexicon_coverage.py` | lexicon coverage tooling |
| `bg_chitanka_clean.py`, `prep_chitanka.py`, `select_chitanka_sentences.py` | Stage 2 corpus prep |
| `mark_chitanka_text.py` | Stage 2 text-only labeling |
| `accentor_model.py`, `train_accentor.py`, `eval_accentor.py` | Stage 3 Accentor |
| `eval_accentor_ood.py` | Stage 3 out-of-domain eval on parliament stenograms |
| `build_stress_candidates.py`, `config/bg/always_final_imperatives.json` | Stage 1 lexicon (always-final override) |
| `mine_homographs.py`, `config/bg/homograph_verbs.json`, `build_accentor_finetune.py` | Stage 6e homograph mining + fine-tune data |
| `vet_stress_cli.py` | Stage 5b interactive stress-label vetting |
| `prep_bg_parliament.py`, `refresh_bg_parliament.py`, `bg_sten_clean.py` | Stage 4 crawl |
| `prep_kaldi_data.py`, `kaldi_pilot.sh`, `harvest_stress_labels.py` | Stage 5 alignment |
| `kaldi_parliament_pilot.sh` | Stage 5b parliament viability pilot |
| `kaldi_milestone_a.sh` | Stage 6a–6c milestone-A scale-up: combined GMM retrain + annotation TDNN + F5 acceptance + harvest |
| `synth_chitanka_bgtts.py` | parked (TTS-distillation only; not for labels) |
| `omnivoice/cli/serve.py`, `serve_bg.sh`, `bg_accent.py` | Stage 7 serving |
| `bon_generate.py`, `bon_judge.sh`, `bon_score.py` | Stage 8 best-of-N oracle harness (generate → align-judge → score) |
| `eval_bg_ab.py` (`--accentor`), `eval_ab_border.txt` | boundary-homograph A/B eval with contextual marks |
| `tokenize_bg400_full.sh`, `train_tts_full.sh` | 413 h TTS scale-up (tokens + launcher) |
| `config/bg/*.json` | curated battery / function words / supplement / train configs |
