# Monetizing OmniVoice-BG ASR — options, prerequisites, risks

**Status: strategy notes, 2026-08-08.** No commitments made. Written to be argued with.

Technical companions: [BG_ASR_BENCHMARK.md](BG_ASR_BENCHMARK.md) (what we can honestly claim),
[BG_ASR_ROADMAP.md](BG_ASR_ROADMAP.md) (what is being built),
[BG_ASR_PUNCTUATION.md](BG_ASR_PUNCTUATION.md) (a prerequisite for most products here).

## 1. What is actually for sale

Not "an ASR model" — three assets, and it is the *combination* that is hard to copy.

| Asset | Concretely | Why it matters commercially |
|---|---|---|
| **Accuracy on Bulgarian** | 4.12 % vs Whisper large-v3's 15.37 % on the same frozen test set | We are not competing with big-tech English quality; we are competing with their **neglect of a long-tail language** |
| **Cost structure** | 73.6 MB int8, CPU-only, ~200 ms for a 14 s clip on one thread | Whisper large-v3 needs a GPU. Marginal cost ≈ electricity. Undercut everyone and still profit at volume |
| **The corpus + BG language stack** | 1,745 h self-built across 8 registers; plus BG TTS, stress front-end, verbalizer, normalizer | The model is a snapshot; **the corpus is the moat**. And ASR+TTS together = a full voice stack, which almost nobody has for Bulgarian |

The deployment story is the differentiator: **an appliance that never sends audio abroad**, not
a subscription to someone else's cloud.

## 2. Plays, ranked

### A. Sovereign transcription for regulated Bulgarian institutions ← recommended start

Courts, municipal councils, parliamentary committees, hospitals, law firms, banks, notaries.

Two of their problems are solved by one product: sending audio to US clouds is a GDPR
headache, **and those clouds are bad at Bulgarian anyway**.

The fit is unusually tight: our model is strongest precisely on institutional/parliamentary
speech, because 573 h of `plsten` is exactly that register. A municipal council meeting is the
same acoustic and lexical world as the training data. **This is the one market where 4.12 % is
an honest quote.**

- Sells as: on-prem appliance or private-cloud instance, per-site licence + support.
- Needs: punctuation, basic diarization, an operator UI. Not phone-band robustness.

### B. The full Bulgarian voice stack — highest ceiling

ASR + LLM + TTS, all Bulgarian, all ours. Call centres, IVR replacement, voice assistants,
in-car, accessibility. GP-S (streaming) lands the missing piece — conversational latency is
impossible with an offline model.

- Sells as: API priced per minute, or per-seat for agent-assist.
- Needs: GP-S, low-latency serving, telephony integration, barge-in/endpointing, diarization.

### C. API + MCP server

An MCP server exposing Bulgarian transcription to Claude and other agent runtimes, with a
plain REST API behind it. Cheap to build — the FastAPI serving patterns already exist in
`omnivoice/cli/serve.py` and `asr_playground.py`.

**Be clear-eyed: MCP is a distribution channel, not a business.** It gets us into agent
workflows for roughly a weekend of work; the revenue is the API behind it. Worth doing
*because* it is small, not because it is the plan.

### D. Media, subtitling, accessibility

Broadcasters, podcasters, YouTube creators; plus public-sector bodies under EU accessibility
pressure to caption. Volume business, price-sensitive — which is exactly where a CPU model
wins. Grant funding is plausible here in a way it is not elsewhere.

### E. Call / meeting recording — the idea with the biggest gap

Real demand, but see §4: we have **zero narrowband training data**, and the legal shape is
consent-first or nothing.

### F. Licensing the model or corpus — hold

Licensing weights to integrators is plausible. Licensing the **corpus** is not, until §5 is
resolved — much of it is material we do not own.

## 3. What each play needs that we do not have

| Capability | A. Institutions | B. Voice stack | C. API/MCP | D. Media | E. Calls |
|---|---|---|---|---|---|
| Punctuation + casing + ITN | **required** | required | expected | **required** | required |
| Diarization (who spoke) | **required** | required | optional | helpful | **required** |
| Streaming (GP-S) | no | **required** | optional | no | required |
| 8 kHz / telephony robustness | no | **required** | no | no | **required** |
| Out-of-domain accuracy evidence | **required** | required | required | required | required |
| Operator UI / workflow | **required** | no | no | required | required |

Two rows deserve attention: **punctuation appears in every column** (see
`BG_ASR_PUNCTUATION.md` — nothing built yet), and **out-of-domain evidence appears in every
column** (see §5).

## 4. The telephony gap, stated plainly

Every hour of the corpus is 16 kHz. Phone audio is 8 kHz, band-limited, codec-compressed, and
often overlapped. On real telephony, 4.12 % could plausibly become 25–35 %. This is the single
largest distance between a demo and a product for plays B and E.

It is fixable and the plan already anticipates it — `BG_ASR_DATA_PIPELINE.md` lists RIR reverb
and "telephony/codec simulation (8 kHz band-limit, µ-law)" as a deferred GP-v3 option. The
cheap first move is augmentation, not collection: downsample a slice of the corpus to 8 kHz and
back, add µ-law/GSM/Opus codec simulation, retrain. Real telephony data would be better and is
much harder to obtain lawfully.

**Do not sell into telephony before measuring on telephony.**

## 5. Risks that bite at month six, not month one

**We cannot yet make a general accuracy claim.** 4.12 % is in-domain parliament. FLEURS/CV are
in-corpus. The first customer who tests us on *their* audio turns that gap into the sales
conversation. This reframes the volunteer playground: it is not a side project, it is **sales
collateral** — the instrument that produces a defensible out-of-domain number.

**Corpus provenance needs a legal opinion before commercialisation.** Parliament stenograms are
public record. BNR archives and YouTube channels are not ours to license. EU text-and-data-
mining exceptions cover a great deal of research use; *selling* a model trained on that
material is a different question, and licensing the corpus is a different question again. One
hour with a lawyer now beats discovering it during due diligence.

**Recording calls is consent territory** (GDPR + ePrivacy). Build consent-first — my-own-calls
dictation, or business call centres that already announce recording — and the consent trail
becomes a *feature* ("auditable, in-country, never leaves your building"). Bolt consent on
afterwards and it is a liability.

**Key-person and reproducibility risk.** The corpus and pipeline live on one workstation. Any
commercial conversation will ask what happens if that disk dies. Backups and a reproducible
rebuild path are a sales asset, not just hygiene.

## 6. Suggested first move

**One pilot with a single Bulgarian municipality or court, for meeting minutes.**

Why this one:
- It is the register where the model is genuinely strongest — the claim is honest.
- The buyer has a budget line and a legal reason to keep audio in-country.
- One public-sector reference customer unlocks the rest of the sector.
- It needs only punctuation + basic diarization + a simple UI — **not** the telephony work.
- It produces exactly the out-of-domain evidence §5 says we lack, and a customer pays for it.

Sequence: punctuation → diarization → operator UI → pilot. GP-S and the arbitration pass
continue in parallel; neither blocks this.

## 7. Open questions for the owner

- Product company, services/integration shop, or licensing play? These need different builds.
- Public sector (slow, sticky, reference-rich) or private (faster, more churn) first?
- Is the corpus an asset to license, or purely internal fuel? §5 changes the answer.
- Does TrainGen remain the anchor customer, with ASR as a second line — or does ASR become the
  main business?
