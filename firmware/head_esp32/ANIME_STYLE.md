# The anime mouth set: style guide

The `anime` style draws the mouth as keyframes (`config/anime/*.svg`, the shape engine of
`src/face_anime.cpp`; the README's "The anime style" section). This page is the pen every
drawing was made with: one line, one palette, one set of sizes, one rule per detail, and the
numbers `pepin.face_style` measures on every keyframe (`tests/unit/test_anime_style.py` fails
on an outlier). A new mouth is drawn in this guide first and in the file second.

The look: classic kawaii manga mouths, drawn after the common anime mouth conventions. A thin
dark-rose line, a flat pink fill, one white band of teeth, round everything; the only sharp
things are fangs.

## The numbers

| rule | value |
| --- | --- |
| canvas | 320 x 170, nothing (strokes included) within 6 px of the edge |
| the mouth's centre | (160, 85) at rest, +-4 px; the thinking lines and struggling sit aside at x 194 |
| line | one width, 3.0 px, for every outline and mouth line (the lip, the band's edge, a bite line, the hooks); 2.0 px for the detail lines only (a tongue's crease, the laugh folds beside a mouth, the shock's stripes); the files' widths are the pen's, the screen's are the style's `line_px` (3.0) and the ratios to the pen |
| corners | nothing sharper than a 5 px radius on fills, outlines and tongues; 2.5 px on the small parts (a highlight, the cheeks' ovals, the moustache's block), and none asked of a band's or a dome's corners on the lip or of a tongue's root under it (within a pixel of the line's edge); fangs are the exception, the only sharp thing. The pen asks 6.5 px of a small bowl's corners and 8 of a big one's: a fillet on a curved edge comes out a little tighter than asked |
| the firmware's polylines | no turn over 35 deg between two of a ring's resampled vertices on a fill or a tongue (a facet shows on a filled edge), 50 deg on a 3 px stroke (its joins are round); the resampling bends to the curves (`face_shapes.resample_bent`), so a corner gets its share of vertices |
| hooks | the ends of a hooked line curl with a 6 px radius through 75 deg, or to the vertical when the end already rises: up on the neutral line, the tease and the cat's, down on sad |
| fill | `#EC7488`; outline `#C04860`; teeth `#FFFFFF`; tongue `#F09AA8`; highlight `#F8C8D0`; cheeks `#F4A6B0`; moustache `#3C2A30`; the card `#FFF5F0` (face.json's `styles.anime.colors`; an expression's `anime.colors` recolour it live: the vampire's red, the terminator's steel) |
| teeth | always one white band under the upper lip, never single teeth: 14 px deep at its middle (11..17), its ends on the mouth's sides a few px under the corners, its lower edge a 3.0 px line that ends on the outline |
| tongue inside a mouth | a plain lighter dome across the bottom (the laugh bowls, happy): no crease, no highlight |
| tongue out of a mouth | one plump rounded tongue (its tip a half circle), its root a pixel under the lip line with its corners rounded (5 px; 5.5 and four pixels in where it lies in front of a lower lip, the lip line stopping under it), drawn with its outline (3.0, the root edge left out), ONE soft oval highlight high on it and the centre crease (2.0 px, from a tenth to two thirds of its height); nothing else carries a highlight |
| sizes | three widths: the closed line 112 px (100..124; its short variant for the thinking lines and the pout 72 px, 64..80), the small open mouth 56 px (46..66), the big open mouth 150 px (134..166); the symbols keep their own: the kiss's "3" 48 px tall, the heart 96 px wide, the idle "e" 34 px tall |
| talking | the upper lip stays put (at the mouth's centre, the half and open drawings' lip within 5 px of the rest's; the corners may round in); a closed line opens into the small bowl (height 0.34 of the width, the band) and its laugh bowl (0.58 of the width, at most 72 px, the band and the dome), both hanging from the line's own curve, shallower under a tongue (0.28 and 0.42, at most 48 px: the tongue then rides on the lower lip, in front of it, drawn in to two thirds and a half); a drawn mouth chews, its height 0.72 then 1.2 of the rest's with the band fixed; a closed drawn shape parts, 1.3 then 1.6 |
| containment | an inner line (a band's edge, a dome's edge, a bite line, a crease) never reaches past the outline: every vertex of it inside the fill or the tongue, at most half a line width out; the outline traces the fill within 1 px, except where a tongue lies in front of the lip |

Measured on the flattened SVG paths (the corner radius: the circumradius over a 2 px window
either side of a point) and on the resampled rings the firmware gets (the turns).

## The mouths

Each SVG's `<title>` describes its drawing. The size class is the first word.

| id | drawing |
| --- | --- |
| neutral | line: the hooked line, its corners a touch up, hooks up |
| smile | line: the wide smile arc |
| happy | big: the bowl, the band, the dome |
| grin | line: the smirk arc, the right end higher, a sliver of teeth under its right half |
| clenched | big: the pill of teeth, the bite line dipping softly at the canines |
| sad | line: a gentle frown, the hooks down |
| worried | line: the upset wave |
| surprised | small: a tall oval with the band, drawn tall |
| thinking | short: the slanted line aside, neutral; its variants `tongue` and `thinking_flat` |
| struggling | short: the thinking line aside, level but for a hint of a smile at its right, the effort tongue curled up out of that corner: concentration, not a smile |
| flat | line: the dead-straight line |
| sleepy | small: the manga set's small bean, tilted, pink, the "z" floating |
| focused | line: the closed lens of teeth with the bite line, its tips soft |
| listening | small: a small soft oval with the band |
| sinister | big: the fang grin, a fang at the raised corner |
| kiss | symbol: the manga set's plump "3", the lower bow the larger; talks as a small oval |
| kiss_heart | symbol: the heart lips, a heart flies |
| pout | short: the bracket pout, a short level line whose ends meet two outward-bowed brackets, the cheeks flushed (the overlay's, at the look's `cheeks` 0.5) |
| thinking_flat | short: the thinking line, nearly level |
| moustache | line: the flat line under the moustache |
| o_mouth | small: the small filled oval |
| vampire | big: the bowl with the band, two fangs hanging from it |
| terminator | big: the wedge of teeth, the bite line and three gaps, in steel |
| blush | small: the small drop between the cheeks (the cheeks are the overlay's) |
| tease | line: the smile with the tasty tongue hanging down out of its right corner, wagging (the `tongue` id's drawing, taken over when that id went on 2026-10-15 night) |
| tired | line: the manga set's tired, a slack tilted lens, the tongue hanging full-width from its lower lip (its root under the lip line, never narrowed), a drop of drool |
| cat, cat_blush | line: the cat's "w", its peak and its ends round (no points on a cat), the cheeks flushed on the second (the overlay's) |
| smile_wide, happy_plain | line, big: the smile's and happy's drawings (the kawaii set's variants; here the same mouths) |
| angry | short: the lens of teeth of focused, shorter, its ends turned down 8 px by a parabola, the bite line |
| shocked | small: the small tall rounded rectangle, 48 x 58, filled, the cheeks and the stripes the overlays' |
| (eps) | symbol: the small "e" the idle gesture shrinks into |
| (cheeks) | the overlay: the hatched blush pair at (160 +-108, 63), drawn over any look at its `cheeks` |
| (stripes) | the overlay: three 2 px lines down the left cheek, drawn at a look's `stripes` |

## Where each mouth lives

The robot's day, by `config/face.json`'s `events`, `config/animation.json`'s compositions and
the brain's face tool (`pepin.tools.face`: any id by name).

- **Resting** (between turns, the idle gestures playing): `neutral`; `smile` while the robot
  speaks (the lip flap on top) and at the first clear face of a conversation.
- **In conversation**: `listening` while the person speaks (small, attentive, calm: it shows
  for whole sentences), `thinking` while the model works (its variants `tongue` and
  `thinking_flat` are drawn at random, so all three must read as one family), the
  compositions' faces as the reply's markers fire: `happy`, `smile`, `grin`, `sinister` (a
  laugh, a greeting, a farewell), `thinking`, `flat` (doubt, a shrug, a no, a stop), `focused`
  (emphasis, a salute), `surprised`, `worried` (sympathy), `clenched` (the evil click),
  `moustache` (a private tree's dance).
- **Driving**: `focused` from the moment Nav2 takes the goal, `struggling` at each recovery (an
  obstacle on the way: a strain with a smile, never a frown; also a VIO restart), `surprised`
  at a stall look, `grin` when the phantom is carved away, `worried` when the obstacle is real;
  the end: `happy` (arrived), `sad` (failed), `flat` (cancelled).
- **Dozing**: `sleepy` when every brain lease lapses and when the head hears nothing for 3 s:
  the face between sessions, restful at a glance, breathing, a "z" now and then.
- **On the brain's word**: the rest (`kiss`, `kiss_heart`, `pout`, `blush`, `tease`, `tired`,
  `o_mouth`, `cat`, `cat_blush`, `vampire`, `terminator`, `angry`, `shocked`, the smile's and
  happy's variants).
