# QFactor-Penny README GIF Design

## Style Prompt

Forensic Quant Cinematic: a dark, precise research-explainer thumbnail with Swiss grid discipline, cinematic depth, and restrained motion. The GIF should feel like a miniature paper abstract: curious, skeptical, and technically polished. It should show the QNN failure mode clearly without looking like a trading dashboard or a sci-fi promo.

## Colors

- `#071210` deep green-black background
- `#eef3e9` off-white paper text
- `#7fc7a6` desaturated mint for benchmark structure and valid audit states
- `#d77768` muted rust for collapse, negative signal, and warnings
- `#e0a856` amber for calendar-heavy features
- `#6d8b97` blue-gray for SPY/reference structure

## Typography

- Display/system sans: Inter, Avenir Next, Helvetica Neue, system-ui
- Mono/data: SFMono-Regular, Menlo, Consolas, monospace
- Use tabular numerals for tickers and metrics.

## Motion

- Use GSAP timelines with labels and transform-only animation.
- Prefer clipped reveals, staggered ticker entrances, score-pulse movement, and smooth bar flattening.
- Use `expo.out`, `power3.inOut`, and `power2.out`; avoid bouncy/comedic motion.
- No infinite repeats inside GSAP. The rendered GIF loop handles repetition.

## What Not To Do

- No neon sci-fi glow, confetti, meme arrows, or generic finance dashboard clutter.
- No dense tables as the main visual.
- No visual implication of QNN success, quantum advantage, trading edge, or real-hardware validation.
- No absolute local paths in source or generated README references.
