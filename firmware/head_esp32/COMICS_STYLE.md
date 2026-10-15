# The comics mouth set: style guide

The `comics` style draws the mouth as keyframes (`config/comics/*.svg`, the shape engine of
`src/face_anime.cpp`; the README's "The anime style" section describes the engine, this set
is its second style). This page is the pen every drawing was made with: one ink line, one
palette, one set of sizes, one rule per detail, and the numbers `pepin.face_style` measures on
every keyframe (`tests/unit/test_comics_style.py` fails on an outlier). A new mouth is drawn
in this guide first and in the file second.

The look: the classic cartoon comics mouth, drawn after the classic-cartoon mouth conventions.
A bold black ink line, the open mouth filled black, a white band of teeth with the gaps drawn
as ink strokes (or two front teeth), one red tongue, white paper. Big and expressive: the mouths
take most of the panel's width.

## The numbers

| rule | value |
| --- | --- |
| canvas | 320 x 170, nothing (strokes included) within 6 px of the edge |
| the mouth's centre | (160, 85) at rest, +-4 px; the thinking lines sit aside at x 194 |
| line | one width, 5.0 px, for every outline, mouth line, tick and tongue outline (a bold brush; at 5 px the small mouths stay open); 3.0 px for the detail lines only: the gaps between teeth, a bite line, the laugh folds, the kiss's seam |
| ink | the line and the mouth's interior are one colour, `#1D1D1B`: an outline round a black mouth only thickens it, and a closed line opens into a black bowl of the same ink |
| palette | fill `#1D1D1B`; outline `#1D1D1B`; teeth and fangs `#FFFFFF`; one red `#DE5B55` for the tongue, the kiss lips, the heart and the cheeks; the drool `#C1D5FA` (a tear blue); the moustache the ink; the paper `#FFFFFF` (face.json's `styles.comics.colors`; an expression's `comics.colors` recolour it live: the vampire's red, the terminator's steel) |
| background | white paper, not the cream card of the other styles: the style is ink on white, the teeth are the paper's white (on cream the teeth would be a second, brighter white), and the black mouth is the screen's own black; the number lives in `styles.comics.colors.background` |
| corners | nothing tighter than a 3 px radius on fills, outlines and tongues (the D laugh's corners, the lens tips, the bowls: the pen asks 4 and 3.5, a fillet on a curved edge comes out a little tighter), 1 px on the teeth (a tooth block's corners sit under the lip), 2.5 on the blush and the moustache; the zigzag (`data-part="zigzag"`, the line and the bowls it opens into) and the fangs are sharp on purpose and not measured |
| the firmware's polylines | no turn over 50 deg between two of a ring's resampled vertices on a fill or a tongue, 60 deg on a 5 px stroke (its joins are round); the sharp parts excepted |
| ticks | the dab beyond each end of a closed mouth line (the style's signature): it starts 8 px past the end (the two round caps apart) and curls through 50 deg with a 12 px radius the way the line bends there, up at a smile's ends, down at a frown's; tagged `tick`, a detail, not part of the mouth's width; they stay while the mouth talks |
| the chin mark | a 22 px arc 11-12 px under the mouth's lowest point (smile, happy, surprised, sleepy), tagged `chin`; left off where it would cross the margin (the open talk drawings of the big mouths) |
| teeth | a white band under the upper lip with no line under it (white meets black), 15..30 px deep at its middle (measured 14..34), its gaps 3 px ink strokes from the lip down to the band's edge (`gap`: one on the laugh and the grin, none on the fanged bands); the two front teeth (grin, listening) a white block hanging from the lip with one gap, tagged `hint` (not the band); the gritted grid (clenched, terminator) is the lens filled white with the bite line and four to six gap strokes over it: the engine's teeth slot holds three polygons, so the grid is one white polygon under black 3 px strokes, within the line slot's eight strokes (the outline, the bite, six gaps) |
| tongue inside a mouth | the red dome across the bottom, no line round it (red meets black), no highlight |
| tongue out | the plump red tongue with its 5 px outline (the root edge left out), its tip a half circle, its root a pixel under the lip (four in front of a lower lip, the lip line stopping under it); no highlight, no crease; the tongue and its outline follow the wag (`data-follow="tongue"`) |
| highlights | none anywhere: a `shine` is a fault; the drool is the anchor's drop |
| fangs | triangles hanging from the band's edge (sinister, vampire), the band itself plain |
| cheeks | flat red ovals, one each side at (160 +-108, 65), drawn soft by the blush slot; no hatching; the overlay `cheeks_rest.svg`, drawn over any look at its `cheeks` (no drawing carries its own); the shock's stripes (`stripes_rest.svg`) three 3 px ink lines down the left cheek |
| sizes | the closed line 100..136 px (the smile 128, comics lines are wide); the short line 60..84 (the thinking lines, the pout); the small drawn mouths 30..90 (the o 36, the sleeper 36, the listener 64, the tired lens 84); the big mouths 146..196 (the D laugh 180, the grin 184, the gritted grid 176, the fanged mouths 150-180); the shout 72..92 wide and taller than wide (80 x 108); the symbols keep their own: the kiss lips 60 wide, the heart smile 72 wide, the whistle curl 38 tall, the idle "e" 30 tall |
| talking | the upper lip stays put (the half and open drawings' lip within 5 px of the rest's); a closed line opens into the small black lens with the red dome (half open, 0.32 of the width) and the D laugh with the band and the dome (open, 0.58 of the width, at most 76 px), both hung from the line's own curve, shallower under a tongue (0.26 and 0.4, at most 44 px: the tongue then rides on the lower lip, drawn in to 0.8 then 0.62 of its width); a drawn mouth chews, its height 0.72 then 1.2 of the rest's with the band fixed, capped at the margin; a closed drawn shape parts, 1.25 then 1.5 |
| containment | an inner line (a gap, a bite line, a seam) never reaches past the outline: every vertex inside the fill or the teeth, at most half a line width out; the outline traces the fill within 1 px, except where a tongue lies in front of the lip |

Measured on the flattened SVG paths (the corner radius: the circumradius over a 2 px window
either side of a point) and on the resampled rings the firmware gets (the turns).

## The mouths

Each SVG's `<title>` describes its drawing; the size class is the first word.

| id | drawing |
| --- | --- |
| neutral | line: the small content smile, a tick at each end |
| smile | line: the wide smile arc, the ticks, the chin mark |
| happy | big: the D laugh hung from a nearly straight lip, the band with one gap, the red dome, the ticks, the chin mark |
| grin | big: the wide thin grin crescent with the two front teeth hanging at its middle, the ticks |
| clenched | big: the gritted grid, a lens of white teeth under the bite line and six gaps |
| sad | line: the drooping frown, the ticks down, the lower lip's short line under it |
| worried | line: the worried wave, two gentle periods, its ends curling up |
| surprised | tall: the shout, a rounded-top U taller than wide, the band with its gap, the dome, the chin mark |
| thinking | short: the lazy S aside, the ticks; its variant `thinking_flat` |
| struggling | line: the tight zigzag, seven straight runs, sharp |
| flat | line: the dead-straight line, the ticks down |
| sleepy | small: the sleeper's small black oval with the red at its bottom, the chin mark, a drop of drool, the "z" floating |
| focused | line: the firm line, a slight frown, no ticks |
| listening | small: the small smile bowl with the two front teeth, the ticks |
| sinister | big: the evil grin, the band with two fangs hanging from its edge, the dome, the ticks |
| kiss | symbol: the red kiss lips, two lobes over a round point, the seam between them; talks as the small black oval |
| kiss_heart | symbol: the small smile with the red heart on it, the ticks; a heart flies |
| pout | short: the small pursed frown, the ticks down, the cheeks flushed (the cheeks the overlay's) |
| thinking_flat | short: the small pursed line aside, nearly level, one tick |
| moustache | line: the flat line under the moustache |
| o_mouth | small: the small black oval |
| vampire | big: the fanged mouth, a wide bowl with the band and two long fangs, the dome |
| terminator | big: the angry gritted grid, tilted, the bite line and four gaps |
| blush | symbol: the whistle's curl between the flushed cheeks (the cheeks the overlay's) |
| tease | line: the smile with the tongue out of its right corner, the ticks; wags (the `tongue` id's drawing, taken over when that id went on 2026-10-15 night) |
| tired | small: the dazed lens with the tongue hanging limp out of its lower lip, a drop of drool |
| cat, cat_blush | line: the cat's w (a wave doubled), its ends curling up; the cheeks flushed on the second (the overlay's) |
| smile_wide, happy_plain | line, big: the smile's and happy's drawings (the kawaii set's variants; here the same mouths) |
| angry | short: the gritted grid of clenched, short (80 px) and its ends turned down 8 px, the bite line and three gaps |
| shocked | small: the small tall rounded black mouth, 40 x 56, the cheeks and the stripes the overlays' |
| (eps) | symbol: the whistle's small curl the idle gesture shrinks into |
| (cheeks, stripes) | the overlays: the flat red pair; three 3 px ink lines down the left cheek |

Where each mouth lives in the robot's day is the same as the anime set's
([ANIME_STYLE.md](ANIME_STYLE.md), "Where each mouth lives").
