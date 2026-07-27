# OmniVoice-BG — Logo & Brand Prompts

Prompts for generating the project logo with **nano banana** (Gemini 2.5 Flash
Image) or any modern text-to-image model. The brand fuses three ideas: **voice**
(waveform / sound), **Bulgarian identity** (Cyrillic + the tricolor), and a
clean, scalable **app-icon** form.

**Current logo:** `deploy/logo.png` (694×694 RGBA) — Concept **A** below: a
Cyrillic **О** dissolving into a Bulgarian-tricolor waveform on a rounded
off-white tile. It is served by the volunteer collector (`omnivoice-collect`)
as the page header image and favicon (`GET /logo`), and is the source for the
README / app icon.

## Palette
Bulgarian flag: **white `#FFFFFF`**, **green `#00966E`**, **red `#D62612`**.
Optional dark ground: **navy `#0E1A2B`** / near-black `#0B0F14`. State the hex
codes in the prompt.

## Tips for nano banana
- Ask for **"flat vector logo, transparent or solid background, centered,
  generous padding."** Logos need negative space.
- Generate the icon at **1:1**; run the horizontal wordmark separately at **3:1
  / 16:9**.
- Text rendering is decent but not perfect — **spell the text exactly, keep the
  font simple, and verify the result** (Cyrillic `БГ` and `OmniVoice-BG` are the
  risky bits). Re-roll if the letters are off.
- Iterate conversationally: "same logo, thicker strokes / remove the gradient /
  simpler."

---

## Icon concepts (1:1)

**A — Waveform monogram (the chosen mark)**
```
Minimalist flat vector app icon for a text-to-speech project. A bold letter "О"
(Cyrillic O) whose right side dissolves into a symmetrical audio waveform / sound
wave. Clean geometric strokes, rounded ends. Bulgarian tricolor accent: white,
green #00966E and red #D62612 used sparingly. Solid soft off-white background,
subtle rounded-square icon container, lots of padding, centered. Modern, crisp,
scalable, no text, no photorealism, flat design.
```

**B — Speech bubble + soundwave**
```
Flat vector logo mark: a rounded speech bubble containing a three-bar audio
waveform, the bars colored in the Bulgarian flag order — white, green #00966E,
red #D62612 — on a dark navy #0E1A2B background. Simple, friendly, geometric,
thick even strokes, centered with padding, transparent-ready, no text.
```

**C — Шевица (Bulgarian embroidery) fusion**
```
Logo icon blending a traditional Bulgarian embroidery "шевица" motif with a
digital audio waveform. Symmetric cross-stitch geometric pattern in red #D62612
and green #00966E on white, where the horizontal middle row morphs into a sound
wave. Flat vector, folk-modern, balanced, square composition, generous margins,
no lettering.
```

**D — Microphone + Cyrillic Г**
```
Clean flat vector emblem: a stylized studio microphone formed from the Cyrillic
letter "Г", with concentric sound rings radiating outward. Two-tone: deep green
#00966E and red #D62612 accents on a light background. Rounded, minimal, tech
startup feel, centered, ample whitespace, no text.
```

**E — Neural voice orb (AI/tech angle)**
```
Modern app icon for an AI voice model: a glowing spherical orb made of flowing
concentric sound-wave rings and subtle neural-network nodes. Gradient from green
#00966E to red #D62612 over a dark #0B0F14 background, soft inner glow, rounded
square icon shape, centered, minimal, futuristic, no text.
```

**F — Bulgarian rose + soundwave**
```
Elegant flat vector logo: a single stylized Bulgarian rose whose petals unfurl
into an audio waveform ribbon. Line-art style, red #D62612 rose with green
#00966E leaf-wave, on cream background. Refined, boutique, symmetric, centered
with padding, no text.
```

---

## Wordmark lockup (wide, ~3:1)

**G — Horizontal logo with Latin name**
```
Horizontal logo lockup for "OmniVoice-BG". Left: a compact icon of the Cyrillic
letter "О" merging into a sound waveform. Right: the wordmark "OmniVoice" in a
clean modern geometric sans-serif, with a small "BG" tag badge in the Bulgarian
tricolor (white/green #00966E/red #D62612). Flat vector, high contrast, on a
transparent/white background, balanced spacing, crisp and legible. Ensure the
text reads exactly "OmniVoice-BG".
```

**H — Fully Cyrillic variant**
```
Wordmark logo reading "ОмниГлас БГ" (Cyrillic) in a bold contemporary sans-serif,
the letter "Г" extended into an audio waveform tail. Green #00966E and red
#D62612 on white, minimalist, flat vector, horizontal, generous kerning, centered.
```
*(“ОмниГлас” = “OmniVoice” in Bulgarian — for a fully Cyrillic identity; drop it
if you prefer the Latin name.)*

---

## Favicon (square, must read at 16–32 px)

The browser-tab icon is a separate, **square** asset — drop it at
`deploy/favicon.png` (256×256 or 512×512) and the collector serves it at
`/favicon`. Favicons need bold shapes and near-zero fine detail. Prompts:

**FAV-1 — Bold monogram tile**
```
Square app favicon, 512x512, for a Bulgarian text-to-speech app. A single bold
Cyrillic letter "О" with a short, thick 3-bar audio waveform inside or beside it,
in the Bulgarian tricolor (white, green #00966E, red #D62612). Solid off-white
rounded-square tile, heavy strokes, very high contrast, large and centered,
minimal detail so it stays legible at 16 pixels. Flat vector, no text, no
gradient, no thin lines.
```

**FAV-2 — Tricolor waveform glyph**
```
Minimal square icon, 512x512: three chunky vertical audio-waveform bars in white,
green #00966E and red #D62612 on a dark navy #0E1A2B rounded square. Bold, clean,
symmetric, thick bars with rounded caps, generous padding, instantly readable at
tiny sizes. Flat vector, no text, no fine detail.
```

**FAV-3 — Green tile + white О-wave (single-color, ultra-legible)**
```
Square favicon, 512x512: a white Cyrillic "О" merging into a white sound wave,
centered on a solid green #00966E rounded-square tile. One-color, maximum
contrast, very bold strokes, lots of padding, crisp at 16–32 px. Flat vector,
no text, no gradient.
```

Tip: after generating, shrink the result to 16×16 and confirm the shape still
reads — if it turns to mush, ask for "thicker strokes, fewer elements, more
padding."

## Workflow
Start with **A** + **G** (icon + lockup in one visual language), pick a
direction, then have the model produce matching variants and export a clean
transparent PNG. Settle on one, then trace the winner into an **SVG** for the
README/favicon. To swap the app logo, replace `deploy/logo.png` (or pass
`omnivoice-collect --logo <path>`).
