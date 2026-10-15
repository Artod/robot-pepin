# The kawaii mouth set: style guide

The `kawaii` style draws the mouth as keyframes (`config/kawaii/*.svg`) with the shape engine
of `src/face_anime.cpp`, beside the anime set and with the same 33 ids: the robot's default
mouth since 2026-10-15, when Artem chose it on the real panel and revised it mouth by mouth.
This page is the pen every drawing was made with, and the numbers `pepin.face_style` measures
on every keyframe against (`uv run python -m pepin.face_style --style kawaii`;
`tests/unit/test_kawaii_style.py` fails on an outlier). A new mouth is drawn in this guide
first and in the file second. The anime guide ([ANIME_STYLE.md](ANIME_STYLE.md)) holds the
engine's rules this one keeps (the canvas, the polylines, the talk keys, containment); here
are the differences.

The look: minimal kawaii mouths drawn after the common kaomoji mouth conventions,
"глупенький". Tiny mouths under big eyes, drawn in one near-black line; a narrow round "u" at rest; the laughs rose inside with one
white outlined tooth; the closed grin and the pillows of teeth with the bite dashed unevenly;
the quiet mouths small ovals and rectangles filled rose; the slack mouths dark; a plain rose
tongue; pink cheeks; a light-blue tear; white paper.

## The palette

The colours (2026-10-15 evening; the afternoon's cocoa read old-fashioned on the panel):

| element | hex |
| --- | --- |
| the line (`lip`, `moustache`), the slack mouths' and the dots' inside | `#504743` |
| the tongues, the laughs' and the quiet mouths' inside (`cavity`), the heart | `#DD7487` |
| the cheeks (`cheek`) | `#EAB1B7` |
| the teeth and the fangs | `#FFFFFF` |
| the teeth's thin outline, the dashes, the stripes | the line's `#504743` |
| the tear (`shade`) | `#90C1E9` |
| the paper | `#FFFFFF` |

## Drawn small, shown large

The files are drawn small, in a 3.5 px pen (2.5 for the teeth's outlines and
dashes), around the set's common centre, and `config/face.json` `styles.kawaii` shows them at
`scale` 2.5 about that centre in a `line_px` of 10 (2026-10-15: the afternoon's proportion
study beside the 27 mm eye rings asked 14, the stroke an eighth of the mouth; on the
panel it read heavy and gloomy, 10 since the evening). The common centre is (160, 70), high
on the panel
(`pepin.face_shapes.centre_of`): the laughs, 100 px tall on the screen, allow no higher. The
numbers below are the files' (the pen's); the check multiplies every length by the scale and
measures the drawing as the screen gets it, so the knobs can change without a redraw, within
the floor and the ceiling: nothing may be drawn lower than y 104 or higher than y 48 in the
file (the screen's 6 px margin and the 10 px line's half width at the scale), which is why
the tallest talk drawings are capped (surprised's and shocked's open frames at 1.5 of their
height, not 1.6) and the bowls under a tongue are shallower.

## The numbers

| rule | value |
| --- | --- |
| canvas | 320 x 170 in the file; on the screen nothing (strokes included) within 6 px of the edge |
| the mouth's centre | (160, 70) at rest, +-4 px; thinking and thinking_flat sit aside at x 194, struggling at x 172 ("set slightly to the right") |
| line | one width, 3.5 px in the file, for every mouth line and outline (the lines, the laughs' and the small mouths' outlines); 2.5 px, the thin pen, for the teeth's outlines and dashes, the band's edge, the fangs' edges, the terminator's bite and gaps and the shock's stripes; on the screen the outline is `line_px` (10) and the thin strokes keep the ratio (7.1) |
| corners | nothing sharper than a 3 px radius on fills, lines and tongues (the tiny ovals' ends, the small bowls' corners, a dashed bite's caps are caps); 2.5 px on the teeth, the cheeks and the moustache; fangs are the exception |
| the firmware's polylines | as the anime guide: no turn over 35 deg between two resampled vertices on a fill or a tongue, 50 deg on a stroke (the resampling's bend scales with the drawing) |
| fills | the laughs (happy, happy_plain, listening, blush, the talk bowls of every closed line) and the quiet mouths (o_mouth, surprised, thinking, worried, shocked) are rose inside, the style's `cavity`; the slack mouths and the dots are dark, the line's own colour: tired, sleepy, thinking_flat and kiss (its talk o) carry `cavity` `#504743` in their `kawaii` blocks of face.json; vampire keeps its anime block's deep red; nothing else is filled |
| palette | the table above (face.json's `styles.kawaii.colors`; the anime blocks' recolours carry over through `like`: the vampire's red, the terminator's steel; the kiss's lip is set back to the line's colour) |
| teeth | white with a thin dark outline: one tooth where the mouth shows one (happy's, 18 x 11 under the top edge, a little right of the middle; listening's, 9 x 8 set right), a white rounded rectangle outlined in the thin pen (`tooth`), its top corners under the outline; the band (10 px deep at its middle, 7..13) with its edge stroked, on vampire alone; the closed teeth mouths (grin, clenched, angry, terminator) are white shapes with the bite across them: grin's D, clenched's pillow (the sides pinched, the whole bent up 3 px at its ends: a hint of a smile, not a smile) and angry's dome (a half ellipse on a flat base) carry the bite dashed, never a mirrored pair: a dash in from the left edge (0.22..0.24 of the span), a short dash off centre (grin and clenched left of it, angry right of it) and a shorter dash in from the right edge (0.14..0.15), the thin pen; the terminator keeps its one bite line and three gaps |
| tongue inside a mouth | the rose dome across the bottom of vampire's dark laugh: no line, no crease, no highlight (the rose laughs need none); no tongue fills a mouth any more (the small mouth the tongue filled did not read as a stuck-out tongue on the robot, 2026-10-15 night: tease took the hanging tongue) |
| tongue out | a plain rose tongue, straight sides and a half-circle tip, its outline drawn (the root edge left out; a root set deep under a lip leaves its corner arcs out too; tired's in the thin pen), no highlight and no crease; it wags (`wag`, the anime blocks': tease 0.5); the effort tongue (struggling) comes out of the line's right end to the right and up, 13 wide and 21 long, and stays there while the mouth talks; tired's lolls down out of the slack mouth's lower right, 16 wide and 22 long, leaning 14 deg |
| cheeks | the overlay `cheeks_rest.svg`: a solid pink oval each side at (160 +-42, 64), 13 x 6.5, `#EAB1B7` (the `cheek` slot: solid, where the anime hatching's ground is soft), drawn over any look at its `cheeks` (blush, pout, cat_blush, shocked); as far out as the screen allows at the scale; no drawing carries its own |
| stripes | the overlay `stripes_rest.svg`: three 2.5 px lines down the left cheek from y 48 to 72, 5.5 px apart, drawn at a look's `stripes` (shocked) |
| sizes | line 34..68 px (the round u 36, the cat's w 46, tease's smile 54, the v 56, the wide u 60); short 20..40 (the tiny u, the n and the pout 28; the firm line and the effort line 30; the dash and the moustache's 34); small 16..44 (the o 20, the shocked rectangle 22, the U 26, the tilted oval 30, listening's laugh 36, blush's 40, the wobbly oval 40, the slack mouth 40, the angry pillow 42); big 60..92 (the laugh 68, the grin's D 68, the teeth rectangles 64 and 68); dot 6..14 (the thinking dot 10, the doze 9); the symbols keep their own: the "3" 40 px tall, the idle "e" 22. Round 2 (the panel, 2026-10-15 evening) narrowed the closed u family: the round u 44 -> 36, the tiny u 34 -> 28, the inverted u and the pout 34 -> 28, the cat's w 56 -> 46 (110 -> 90, 85 -> 70, 85 -> 70 and 140 -> 115 px on the screen); the laughs kept their width |
| talking | the upper lip stays (the talk drawings' lip within 5 px of the rest's); a closed line opens into a small dark bowl (0.42 of the width, at least 18 px deep) then the laugh bowl (0.7 of the width, at least 26) with the red dome, both hung from the line's own curve (a narrow line gets 3.4 px corners and a frown 5 px ones; a line whose ends rise steeper than 48 deg, the round u, is hung 3 px wider a side and its bowl bulges 0.12 of the width with 4.5 px corners, or the fillets would eat its steep ends; the cat's w opens into a bowl hung from a plain arc through its ends); a drawn laugh chews, 0.72 then 1.2 of its height with the top fixed; a closed shape parts, 1.3 then 1.6 (1.5 where 1.6 would cross the floor); the "3" parts into a small dark o under its top; a tongue out hangs from the bowl's lower lip while the mouth talks, drawn in (0.85 and 0.7 of its width, 0.5 and 0.4 of its length), the bowl shallower to leave it room above the floor; the effort tongue stays at its corner |
| containment | as the anime guide: an inner part never reaches past the outline (half a line width at most), the outline traces the fill within 1 px but under a tongue |
| drops | none since the revision (sleepy's and tired's went) |
| effects | the z (sleepy), the heart (kiss_heart), the wag (tease 0.5), no crawling wave on struggling (its `kawaii` block) |

Measured on the flattened SVG paths and on the resampled rings the firmware gets, as the
anime set is (`pepin.face_style.KAWAII`), after the scale.

## The mouths

Each SVG's `<title>` describes its drawing. The size class is the first word.

| id | drawing |
| --- | --- |
| neutral | short: the tiny u |
| smile | line: the round u, the resting face |
| happy | big: the laugh, flat on top and round below, rose inside, one outlined tooth a little right of the middle |
| grin | big: the closed grin, a D of white teeth, the bite dashed unevenly |
| clenched | big: the pillow of teeth, slightly upturned, the bite dashed unevenly |
| sad | short: the inverted u |
| worried | small: the wobbly oval, rose |
| surprised | small: the small U, a flat top and a round bottom, drawn taller, rose |
| thinking | small: the tilted small oval, rose, aside |
| struggling | short: the short line aside, tilted, a hint of a smile, the tongue out to the right and up |
| flat | short: the dash |
| sleepy | dot: the dot of a doze, the z floating |
| focused | short: the firm line, its ends a hair up (round 2: turned down it read sad) |
| listening | small: the small rose laugh with one outlined tooth, set aside |
| sinister | line: the v tilted, its right end higher, a sharp fang under that arm, 14 x 22 |
| kiss | symbol: the "3"; talks as a small dark o |
| kiss_heart | symbol: the "3", a heart flies |
| pout | short: the small n frown, the cheeks the overlay's |
| thinking_flat | dot: the dot, aside |
| moustache | short: the dash under the moustache's block |
| o_mouth | small: the small o, rose |
| vampire | big: the dark laugh with the teeth band and the rose dome, two fangs hanging from the band's lower edge (the upper gum; the afternoon's hung from the lip by a slip of the pen), the band's edge and the fangs' edges outlined in the thin pen so the white reads on white |
| terminator | big: the rounded rectangle of teeth in steel, the bite line and three gaps |
| blush | small: the small rose laugh between the cheeks, the cheeks the overlay's |
| tease | line: the shallow smile with the tongue hanging from its middle, wagging (the `tongue` id's drawing, taken over when that id went on 2026-10-15 night: the small open mouth the tongue filled did not read as a stuck-out tongue on the robot) |
| tired | small: the dark slack mouth, an arch over a flatter bottom, tilted 8 deg, the rose tongue lolling down out of its lower right in the thin outline (the "x x" eyes are the rings') |
| cat, cat_blush | line: the w, the cheeks the overlay's on the second |
| smile_wide | line: the wide u smile: the smile of the morning, kept as its variant |
| happy_plain | big: the laugh without the tooth: the laugh of the morning, kept as happy's variant |
| angry | small: the pillow of gritted teeth under an arch, the bite dashed unevenly (the brows are the rings') |
| shocked | small: the tall rounded rectangle with a waist, rose; the cheeks and the stripes the overlays' |
| (eps) | symbol: the "3" mirrored and small, the e the idle gesture shrinks into |
| (cheeks) | the overlay: the solid pink pair, `#EAB1B7` |
| (stripes) | the overlay: the three stripes down the left cheek |

Where each mouth lives in the robot's day is the anime guide's last section: the same ids, the
same events; `angry` and `shocked` are the brain's to ask for (`face:angry`, `face:shocked`),
and the book's pools draw `smile_wide` and `happy_plain` beside their bases.
