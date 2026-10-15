# Head firmware (ESP32): the mouth and the head IMU

The robot's head is a stereo camera on a pan/tilt neck. Its two lenses are the eyes; under them
sits a 1.9" ST7789 screen (320x170, landscape) that is the mouth, driven by an ESP32 board that
also samples an MPU6050 glued rigidly to the camera body. The ESP32 talks to the robot's board
over USB serial only; WiFi and Bluetooth are never started.

```
board (Orange Pi)                                    head
pepin.head_server ── /dev/pepin-head (CH340, 921600) ── ESP32 ── SPI 80 MHz ── ST7789 (mouth)
  TCP 3340: face commands in, IMU lines out                    └─ I2C 400 kHz ── MPU6050 (head IMU)
```

## Hardware

**Board:** ideaspark ESP32 development board, ESP32-WROOM (classic), 16 MB flash, integrated
1.9" ST7789 170x320 TFT, CH340 USB-UART, USB-C.

**Display pins (on the board, fixed):** MOSI 23, SCLK 18, CS 15, DC 2, RST 4, backlight 32 (PWM).
These are VSPI's native pins, so the panel runs at 80 MHz without the GPIO matrix.

**MPU6050 wiring (GY-521, 3-5 cm of wire):**

| GY-521 | ESP32 board | Note |
| --- | --- | --- |
| VCC | 5V (VIN) | the GY-521 has its own 3.3 V regulator and pull-ups to it; 3V3 also works |
| GND | GND | |
| SDA | GPIO 21 | the Arduino core's default I2C pins, free on this board |
| SCL | GPIO 22 | |
| INT | GPIO 27 | optional but recommended: exact sample stamps; without it the firmware polls |
| AD0 | open | pulled low on the module: address 0x68 (0x69 is tried too) |
| XDA, XCL | open | |

Glue the GY-521 flat and rigid to the camera body (not to the screen or the bracket), and note
which way its printed X and Y arrows point relative to the camera: the head IMU's mount is a
number the visual-inertial side needs.

## Build and flash

PlatformIO needs no install beyond `uv`:

```bash
cd firmware/head_esp32
uvx --from platformio pio run                                    # build
uvx --from platformio pio run -t upload --upload-port /dev/cu.usbserial-XXXX   # flash
```

The board enters its bootloader by itself (DTR/RTS auto-reset). Flash from the laptop's USB, then
plug the board into the robot. The serial line carries binary frames, so a serial monitor shows
noise; use `pepin.head_server` (below) instead.

**Bench check on the laptop** before the robot: run the head server against the board's port and
talk to it with the CLI:

```bash
uv run python -m pepin.head_server --config config/head.json --device /dev/cu.usbserial-XXXX
uv run python -m pepin.head_link --host 127.0.0.1 status        # link, fps, IMU rate, clock
uv run python -m pepin.head_link --host 127.0.0.1 express happy 3
uv run python -m pepin.head_link --host 127.0.0.1 show 'Temps|left: 41/70 C' 6
uv run python -m pepin.head_link --host 127.0.0.1 style anime   # the face's style, live
uv run python -m pepin.head_link --host 127.0.0.1 imu 5         # samples in m/s^2 and rad/s
```

## The mouth

The mouth is procedural: 34 numbers draw it (`config/face.json`, `params`): width, opening,
corner lift, how the opening splits above and below the mouth's line, the opening's profile (a
lens, an ellipse or a rounded rectangle), teeth (none, the upper row, clenched), a wave along it
(a worried wobble or a crawling zigzag; a standing wave settles to its home phase, so the cat's
"w" is a "w"), asymmetry, position, line thickness, two idle motions (breathing, drift), a
Cupid's bow in the upper lip, and the decorations: the tongue hanging out over the lower lip (how
far, where along the mouth, how far its tip leans aside), the blush on the cheeks, folds round
the corners (dimples), a short block moustache, two fangs (or one: `fang_side`), the kiss's pucker
(the line shrinks into a "3"), corner hooks (the line's ends curl down or up), heart-shaped lips
(the mouth a filled heart), a dumbbell profile (`bulb`: lips pressed into an "mm"), a drop of
drool, the tongue patch inside an open mouth (`patch`), the tongue's wag, where the fangs
hang (`fang_at`) and the shock's stripes down one cheek (`stripes`: the shape engine's overlay,
this engine draws none). Each element (lip, cavity, teeth, tongue, cheek, moustache, heart, shade) has its own
colour per expression, the defaults being the cyan face's (`colors`). An expression is a full
set of these numbers and colours, and the firmware tweens every one on its own (smoothstep, 280
ms by default), so any expression turns into any other.
The face keeps `margin_px` (6) inside the screen: an opening that would reach past it (speech on
a wide-open smile) is scaled down to fit, the tongue and the drool are shortened, the heart is
held in.

**The styles, one table** (`config/face.json` `style`, `styles`: the block's order is the
'C' config's byte and the firmware's `Style` enum). The same ids and names draw several ways:
`pepin` (0), the thick cyan line on the black screen; `rose` (1), the pepin shapes as line art
on a cream card (the first manga try of 2026-10-14, kept); `manga` (2), the anime conventions
bent into the parametric line; `anime` (3), mouths drawn after the common anime mouth
conventions by a second engine (below); and the styles drawn like it after them: `kawaii` (4),
tiny mouths after the common kaomoji conventions (below), and `comics` (5), mouths after the
classic-cartoon conventions in the same engine ([COMICS_STYLE.md](COMICS_STYLE.md): a 5 px ink
line, black mouths, white teeth with the gaps drawn as ink strokes, one red, white paper; a
proof of concept with the same 33 ids, so the sets compare side by side). Every expression has a look per style: its own numbers and colours in `pepin`, and in a later style the same numbers with
that style's block (`rose`, `manga`, `anime`: `params`) over them and that style's palette
with the block's `colors` over it (`like` carries the blocks). A style sets the line's
thickness range, whether the teeth are one band and how far it reaches, the tongue's width
and highlight, the shade band, the outlines round the decorations, the pucker's stroke, the
lip flap's shape count and the background (`styles`). A style `like` an earlier one starts
from it (its numbers, palette, engine and own keys, and in every expression and idle gesture
its block under this style's own; it plays the idle gestures the liked style plays), so a new
set of drawings registers as one `styles` entry (`engine` shapes, `like` anime, its palette)
plus its directory of keyframes, nothing else. The switch is live: the head server sends
`face.json`'s `style` as the last byte of every `'C'` config, and `{"cmd": "config", "style":
"anime"}` on its door (`pepin.head_link style anime`) changes it on the head without a
rebuild; the face shown tweens into its look in the new style (between two drawn styles, the
drawings morph), the variant drawn and a heart in flight stay. The simulator has the same
switch (a select over the face) and `preview.js` takes `--style`.

## The anime style: a shape-keyframe engine

The parametric mouth bends one line, so every style it draws is that line in another palette.
Anime mouths (a cloud of a sob, a filled drop, a bowl with a teeth band and a plump split
tongue, a grey "mm" dumbbell, a heart, an "ε") are silhouettes, and the `anime` style draws them
with a second renderer (`src/face_anime.cpp`, `sim/anime.js`) beside the parametric one; the
three older styles are untouched and stay selectable.

**Keyframes.** Every expression is three drawings, `config/anime/NAME_rest.svg`, `NAME_half.svg`
and `NAME_open.svg` (at rest, half open, open: the serifu lip flap's three), on a 320x170
canvas. A drawing is closed outlines in layers, each path's `class` naming its layer: `fill`
(the mouth's interior), `teeth`, `tongue`, `shine` (a highlight), `line` (the outline strokes,
the generator draws them `line_px` wide, a thinner stroke keeping the file's ratio to the
set's pen: a 2 px gap in a 3.5 px set is `line_px * 2 / 3.5`), `blush` and `cheek` (the soft ellipses and the strokes
of a blush), `fang`, `moustache` and `drool` (a circle: where a drop hangs and how big). The
layers draw in that order, each in its element's colour from the table (`lip` the outline,
`cavity` the fill, `shade` the highlights and the drool), so an expression's `anime` block
recolours it as in the other styles (sleepy's grey, the vampire's red, the terminator's steel).
`pepin.face_shapes` flattens the paths, resamples every polygon of a layer to that layer's fixed
vertex count (48 for a fill, 48 for an outline, 12-32 for the small parts) starting at its
leftmost point and running clockwise, at steps that count each radian of turning as 16 px of
path (`resample_bent`: a rounded corner gets its share of vertices instead of an 8 px chamfer),
turns a closed stroke into its ring and an open one into a line drawn there and back (so a
closed line opens into a bowl's outline), and writes one table per drawn style,
`src/<style>_table.h` and `sim/<style>_table.js` (a `ShapeTable` in its own namespace: the
keyframes, polygons, vertices, gestures and the style's numbers), the layout they share
(`src/shape_layout.h`: the layers and their vertex counts, the units, the structs) and the
indexes the engine picks a table from by the style in force (`src/shape_tables.h`,
`sim/shape_tables.js`; `face_anime.h` `shape_table_of`, `anime.js` `tableOf`), all by
`uv run python -m pepin.face --write` with the parametric tables; the tests refuse stale
copies. The anime table holds 16,963 vertices (int16, 1/16 px): 68 KB of flash for the 102
drawings; the comics table 22,470 vertices, 90 KB (its ticks, gap strokes and chin marks are
rings of their own); the kawaii table 12,628, 51 KB; a second drawn style costs its own table
and nothing in RAM (the engine's sets are sized by the shared layout).

**Scale and line.** A style's drawings have their own size and the screen shows them at the
style's `scale` (`styles.<name>.scale`, 1 by default), every shape multiplied about the set's
common centre (`pepin.face_shapes.centre_of`: the screen's middle, or the set's own, the
kawaii mouths' (160, 70)) before resampling; the strokes do not scale, `line_px` sets the
outline's width on the screen and the files' ratios the rest. The style check measures the
scaled drawing (sizes, the margin, the stroke) against the guide's lengths scaled the same way,
and a scaled keyframe that reaches past the 6 px margin is refused by the generator.

**Overlays.** Two extra rest keyframes of no expression are drawn over any look, as they are,
at the opacity of a parameter of the same name: `cheeks_rest.svg` (the style's cheek pair:
kawaii's solid pink ovals, anime's hatched blush, comics' flat red ovals) when the look's
`cheeks` is above 0, `stripes_rest.svg` (the shock's three vertical lines down the left
cheek) at its `stripes`. The table lists them in `ShapeTable::overlays` (`shape_layout.h`
`kAnimeOverlayParam` names the parameter of each; the simulator's table carries
`overlays: [{name, frame}]`), the engine draws them after the mouth's layers without the
gestures' and the breath's deformation, and they need no outline stroke. `blush`, `pout`,
`cat_blush` and `shocked` carry cheeks by their own `cheeks` (0.5 to 1), `shocked` the stripes;
the parametric engine draws its own cheeks and no stripes. So no drawing carries its cheeks
any more, and a later `{cheeks}` mark can flush any face.

**The drawings** follow one style guide, [ANIME_STYLE.md](ANIME_STYLE.md): one 3 px line and
one palette, nothing sharper than a 5 px radius but the fangs, one 14 px band of teeth, three
mouth widths on one centre, highlights on the tongues out and nowhere else, no inner line past
the outline, the upper lip fixed while the mouth talks. `pepin.face_style` measures every
keyframe against it (`uv run python -m pepin.face_style`), and
`tests/unit/test_anime_style.py` fails on an outlier: a new mouth is drawn in the guide first.

**Morphing.** The model (`AnimeModel`) follows the parametric model: an expression change, a
style switch or a talk cut (the speech level cut into the three drawings exactly as the manga
flap cuts it) is a tween from the shape on the screen to the target drawing, vertex by vertex
with the same smoothstep (`transition_ms`; a talk cut `styles.anime.talk_tween_ms`, 60 ms). A
layer one side lacks is the other side's polygon collapsed onto the lacking drawing's outline
(the nearest points of its `line` stroke), fading, so a bowl's fill grows out of the closed
line and a tongue shrinks back into the lip. An intensity under 1 blends the drawing with the
neutral rest drawing the same way. The idle gestures of `config/face.json` play in shape space
through their `anime` block: a shift, a scale about the mouth's centre, a tilt, a corner curl
(the twitch) and a morph toward another drawing (the grin toward the smile, the "ε" toward
`config/anime/eps_rest.svg`), with a deep breath of this style alone; the continuous breath,
the drift and the crawling wave (`breathe`, `drift`, `wave` of the expression's `anime`
`params`) deform the vertices, the tongue's `wag` shears the tongue and the strokes marked
`data-follow="tongue"`. The effects: the kiss's heart flies as in the other styles; the sleeper's
`z` (the `z` effect, `timing.z_s`, `z_every_s`) floats up from the mouth's corner again and
again while sleepy shows; the drool's drop stretches down and snaps back every `drool_s`.

**Rasterizing.** Strokes become polygons on the head (the centreline offset to each side, mitres
on gentle turns, round joins on sharp ones, round caps where a line turns back), and every
layer's polygons are filled by scanlines with anti-aliased coverage: the signed area of each
edge is accumulated along the row and the running sum's magnitude, clamped to 1, is the
coverage (the accumulation rasterizer of font renderers). The fill rule is non-zero winding:
a stroke's two sides, run in opposite directions, fill as a band, and overlapping parts of a
morph in flight fill rather than flicker. The frame goes into the same two DMA strips as the
parametric face. Cost on this computer (scalar builds, every expression at rest and talking):
the parametric manga face 24 µs a frame, the anime face 31 µs (1.3x; its worst frames, a
retarget's collapse onto the outline or the widest bowl, 2x); the ESP32 at 240 MHz runs the
same float code some 80-100x slower, so the anime frame is about 3 ms of its 20, within the 50
fps. RAM: 44 KB more than the parametric firmware (the shape sets, the frame's polygons, the
edge buffers; 98.9 KB static in all), flash 100 KB more (1.33 MB in all with the three drawn
sets).

**What each id draws** in the anime set (the comics set's are in
[COMICS_STYLE.md](COMICS_STYLE.md); a drawing's `<title>` describes it; the guide's table has
the size class of each):

| id | anime drawing |
| --- | --- |
| neutral | the hooked line, hooks up |
| smile | the wide smile arc |
| happy | the bowl with the band and the dome |
| grin | the smirk arc, the right end higher, a sliver of teeth under its right half |
| clenched | the pill of teeth, the bite line dipping at the canines |
| sad | a gentle frown, the hooks down |
| worried | the upset wave |
| surprised | the tall oval with the band |
| thinking | the short slanted line aside (variants as before) |
| struggling | the smile with the effort tongue curled up out of its corner |
| flat | the flat line |
| sleepy | the small closed pill with the band and a drop of drool, a z floating up |
| focused | the closed lens of teeth with the bite line, its tips soft |
| listening | a small soft oval with the band |
| sinister | the fang grin |
| kiss | the plump "3" (talks as a small oval) |
| kiss_heart | the heart lips, the heart flying off |
| pout | the small frown, the hooks down, the cheeks flushed |
| thinking_flat | the thinking line, nearly level |
| moustache | the flat line under the moustache |
| o_mouth | the small filled oval |
| vampire | the bowl with the band and the dome, two fangs hanging from the band |
| terminator | the wedge of teeth, the bite line and three gaps, in steel |
| blush | the small drop between the cheeks |
| tease | the smile with the tasty tongue hanging down out of its right corner, wagging (the tongue id's drawing, taken over on 2026-10-15 night) |
| tired | the slack low mouth, the tongue lolling down over the lower lip, a drop of drool |
| cat, cat_blush | the cat's w with its hooks up (the cheeks the overlay's on the second) |
| smile_wide, happy_plain | the smile's and happy's drawings (the kawaii variants are theirs alone) |
| angry | the lens of teeth, short, its ends turned down, the bite line |
| shocked | the small tall rounded mouth, the cheeks and the stripes the overlays' |
| (cheeks, stripes) | the overlays: the hatched blush pair; three thin lines down the left cheek |
| (eps) | the small "e" the idle gesture shrinks into |

The closed lines talk as the small bowl with the band (half open) and the laugh bowl with
the band and the dome (open), both hung from the line's own curve; a tongue out rides on the
lower lip while the mouth talks; the drawn mouths chew (the jaw raised to two thirds, then
dropped past the rest, the band staying put).

**The kawaii set** (`kawaii`, style 4; `config/kawaii/*.svg`) is the robot's default mouth since
2026-10-15 (Artem's choice on the real panel) and a second set for the same engine and the same
33 ids: minimal mouths after the common kaomoji conventions, revised on the panel: the narrow
round "u" at rest, the laughs rose inside with one white outlined tooth, the closed grin and the
pillows of teeth with the bite dashed unevenly, the dot of a doze, the tongues plain and rose,
the quiet mouths rose ovals and rectangles, the slack mouths dark, one near-black line, pink
cheeks, white paper (the guide lists the palette's hexes). The drawings are small (a 36 px u in
a 3.5 px pen) and the screen shows them at `scale` 2.5 about their common centre (160, 70), high
on the panel, in a 10 px line (14 in the afternoon's proportion study beside the 27 mm eye rings
read heavy and gloomy on the panel). Its guide: [KAWAII_STYLE.md](KAWAII_STYLE.md);
`pepin.face_style --style kawaii` measures it against it. It registered as one `styles` entry
(`like` anime, `shapes` kawaii, its palette, its scale and line; the dark interiors and the
kiss's line as `kawaii` blocks on five expressions) and its directory; its table costs 51 KB of
flash and no RAM.

**The manga style** follows the standard anime mouth conventions as the drawing guides teach
them, not the pepin curves:

- the open mouth is "the lower half of an inverted egg" under a horizontal upper lip, the tongue
  "a line that curves upward at the bottom of the mouth" from one side to the other, the teeth
  "a line that curves downward at the top" that "shouldn't have any contact with the sides", the
  inside darkest, the tongue lighter, the teeth white ([Corel Painter, The Complete Guide on How
  to Draw Anime Mouths](http://www.painterartist.com/en/tips/draw-anime/mouths)): `happy`,
  `grin` and every bowl a closed line opens into (`upper` 0,
  `teeth` one band of `teeth_span` 0.85, `patch` the tongue curve);
- when the mouth opens "the top row of teeth stays in the same position" and "the bottom row
  shifts along with the jaw"; teeth are "one combined shape"; the one-fang look "shows only the
  one sharp tooth (like a fang) to one side of the mouth" ([AnimeOutline, How to Draw Anime &
  Manga Teeth](https://www.animeoutline.com/how-to-draw-anime-teeth-tutorial/)): the bowl's
  flat top with the band, `sinister`'s fang hanging from the band on the raised side;
- the relaxed mouth "a light curve", the smile "a bigger curve" with the corners up, the curious
  mouth "a rectangle with rounded corners... top row of teeth slightly visible" (`listening`),
  the puzzled mouth an "O" (`o_mouth`), the sly mouth "wave-shaped... raised toward the corners
  and again toward the middle" (`cat`), the grin "tilted to one end and with the upper lip
  raised higher up" with "a sharp tooth for a malicious look" (`sinister`), the upset mouth "a
  series of waves", not symmetrical (`worried`), the embarrassed mouth slightly open and a
  little wider at the bottom (`blush`), the yelling mouth's inside darkest, tongue lighter,
  teeth white ([AnimeOutline, How to Draw Anime and Manga Mouth
  Expressions](https://www.animeoutline.com/?p=551));
- the tongue sticks "upwards and to one side" out of a mouth "wider on the side where the tongue
  is sticking out" (`tongue`), the playful tease "a wave like shape (looking a bit like a cat's
  mouth) and the tongue sticking out downwards" (`tease`), a split down the tongue (the crease)
  ([AnimeOutline, How to Draw Anime Tongue Out Face](https://animeoutline.com/?p=8817));
- surprise runs "from closed mouths... to O-shaped mouths to fully-open mouths with teeth and
  tongue visible" (`surprised`: the tall oval with the band and the tongue), "sad mouths tend to
  get more open the greater the emotion" (`sad`: the downturned open mouth), the tongue hanging
  limp out of a slack mouth (`tired`), fangs for a snarl
  ([Clip Studio Tips, Drawing Anime Mouth
  Expressions](https://tips.clip-studio.com/en-us/articles/6752));
- the lip flap (serifu) is three drawings, "closed, half open, and open", arranged over the
  dialogue ([What is SERIFU? Animating lip sync action in
  Anime](https://videohighlight.com/v/SMVSJCcmBDU)): `flap_shapes` 3, the speech level cut into
  the three, no in-between;
- the corner hooks on the closed lines, the "3" kiss, the "ω" cat with hooks, the filled heart,
  the lens of teeth, the small bean of `sleepy` and the plump tongue with its highlight are the
  common anime conventions, the hooks and the "3" the kaomoji ones (`:3`, `ω`); `pout` and `thinking_flat` keep the rose shapes (Artem 2026-10-15).

Its animation, in that style: **talking** is the lip flap, the hooked line cut to the half bowl
and the full bowl as the level crosses a quarter and three quarters, the top edge and its band
staying put while the jaw drops and the tongue curve rises with it (`flap_shapes`, the latent
`upper` 0, `teeth` and `patch` of every closed look); **the laugh** (`happy`) breathes its bowl's
height (`breathe` 0.8: +-4 px at 0.25 Hz); **the kiss** puckers the line into the plump "3" over
the transition and `kiss_heart` pops the heart off the heart-shaped lips; **the tongue** slides
out over the transition and wags at `wag_hz` (`wag`); **the sleeper** sends a "z" floating up
and away at once and another every `timing.z_every_s` while it shows, each growing as it rises
and fading over `timing.z_s` (the `z` effect in the manga block's `effects`: the pepin and rose
sleepers have none); **the idle** of the resting faces adds two
gestures of this style alone (`idle.gestures` `styles`): a corner **twitch** (the hooks flip for
a third of a second) and a tiny **"ε"** the line shrinks into for most of a second, beside the
grin, smirk, hmm, tilt and blink every style makes.

The ground is the cream card, as rose's (`styles.manga.colors.background`; Artem 2026-10-15);
the screen's black is the same number changed, and `manga_all.png` shows both.

**At rest the face is alive** (`config/face.json`, `idle`): while a resting expression shows
(neutral, smile), every 8-20 s one gesture passes over it as a smooth bump: a brief wider smile, a
small tilt either way (asymmetry and a sideways shift) or a blink-like thinning of the line. None
starts within 3 s of an expression change or of a speech level, and one under way fades out in
0.25 s when an event or speech comes. The choices are random (seeded from `esp_random()` at
boot); the simulator draws the same ones from the same seed.

Expressions, by wire id (append only):

| id | name | looks like |
| --- | --- | --- |
| 0 | neutral | a short, slightly lifted line |
| 1 | smile | a closed smile |
| 2 | happy | an open D-shaped smile, upper teeth |
| 3 | grin | a wide grin, upper teeth |
| 4 | clenched | a grin with clenched teeth |
| 5 | sad | corners down |
| 6 | worried | a wobbly, slightly open frown |
| 7 | surprised | an O |
| 8 | thinking | a short skewed line off to the side, drifting |
| 9 | struggling | a crawling zigzag (anime: the smile with the effort tongue curled up out of its corner) |
| 10 | flat | a dead-straight line |
| 11 | sleepy | a small round mouth breathing slowly (manga: the small bean, a "z" floating up now and then; anime: the small grey pill with its band and a drop of drool, a z floating up) |
| 12 | focused | a short firm line |
| 13 | listening | a small open, attentive mouth |
| 14 | sinister | a wide lopsided grin with a sawtooth bite |
| 15 | kiss | pink lips pursed into a "3" |
| 16 | kiss_heart | the kiss, and a heart flies off to the side and up |
| 17 | pout | small thick lips pushed out in a frown, flushed cheeks |
| 18 | thinking_flat | a short, nearly straight line off to the side |
| 19 | moustache | a short dark moustache over a narrow, nearly straight mouth |
| 20 | o_mouth | a small round O |
| 21 | vampire | an open red smile with two fangs |
| 22 | terminator | a clenched steel grin, red behind the teeth |
| 23 | blush | a small wobbly mouth between hatched pink cheeks (kawaii: the small laugh) |
| 24 | tease | a short side line, the tongue hanging out towards one end, wagging (the old `tongue` face: the drawn styles' smile with the tongue out of its corner, kawaii's out of its middle) |
| 25 | tired | a slack open mouth, the tongue hanging limp out of one side (rose: the grey dumbbell "mm"; manga: the plump tongue lolling out; anime: the tongue lolling down over the lower lip and a drop of drool; kawaii: the dark slack mouth, the tongue out of its lower right) |
| 26 | cat | the anime cat's "w" mouth |
| 27 | cat_blush | the cat's "w" between blushing cheeks |
| 28 | smile_wide | the smile's wide variant (kawaii: the wide u; the same look as smile elsewhere) |
| 29 | happy_plain | happy's toothless variant (kawaii: the plain laugh; the same look as happy elsewhere) |
| 30 | angry | a tight downturned mouth of gritted teeth (kawaii: the small pillow of teeth under an arch) |
| 31 | shocked | a small tall rounded mouth open in shock, the cheeks flushed, three stripes down one cheek (the drawn styles' overlay) |

(2026-10-14 night: the lipstick print's `kiss_mark` and `kiss_full`, ids 29 and 30, went; tease
and tired moved up and the cats came after them. 2026-10-15: `sinister_red` (22) and the three
dimpled faces (26-28) went after the kawaii review; vampire to cat_blush moved up, the smile's
and happy's variants and the two new faces came after them. 2026-10-15 night: `tongue` (18)
went, `tease` took its drawing and its numbers; thinking_flat to shocked moved up one. The
firmware and `config/face.json` go to the head and the board together: an older firmware would
draw the wrong face for ids 18-31.)

An expression may name **variants** (`thinking`: `tongue`, `thinking_flat`): each time it is asked
for, the firmware shows it or one of them, alike at random; the status line still names the one
asked for, so the head server sees no mismatch. **Effects** are overlays a look carries (`effects` on
the expression, or in a style's block for that style alone): `heart` flies once each time the
expression is set, up from beside the mouth, fading over `timing.heart_s` (a resend while it
flies does not restart it); `z` sends a "z" up at once and again every `timing.z_every_s` while
the look shows, each flight `timing.z_s` (a fixed cadence; a change to a look without it ends it
at once). The model runs one z for both engines: the parametric styles draw it as a three-stroke
glyph (`geometry.z_glyph_px`, `z_glyph_rise_px`), the anime engine as its own smaller one.

Rendering is column by column: for each screen column the lip curves (sampled every half
pixel) are swept with a disc of the line's radius, which gives round ends and an even line on
steep parts, and each column becomes at most four vertical spans (lip, cavity, two rows of teeth)
with anti-aliased ends; the decorations are more such spans (at most 768 a frame) drawn under the
mouth (the blush) or over it (the tongue, the drool, the fangs, the folds, the hooks, the
moustache, the pucker, the heart-shaped lips, the heart). The frame is drawn into two 320x85
strips (2 x 54 KB of DMA-capable RAM)
and each strip is pushed to the panel by DMA while the other is drawn. The face is capped at
50 fps; the frame rate is in the status line.

**Host silence:** with no frame from the board for 3 s (the head server pings at 2 Hz) the mouth
falls asleep (`sleepy`) and wakes into the board's expression when it speaks again. An info screen
('T') covers the face for its seconds.

`src/face_table.h` and `sim/face_table.js` are generated from `config/face.json`, and the
shape engine's tables (`src/shape_layout.h`, `src/<style>_table.h`, `src/shape_tables.h`,
`sim/<style>_table.js`, `sim/shape_tables.js`) from `config/<style>/*.svg`:

```bash
uv run python -m pepin.face --write     # after editing config/face.json or a keyframe
```

## The simulator

`sim/index.html` runs the same model and renderers (line-by-line ports in `sim/face.js` and
`sim/anime.js`, checked pixel for pixel against the C++ by `tests/unit/test_face_parity.py`) with the head server's
arbiter, robot-event buttons, lip sync from a synthetic babble, the microphone or an audio file,
the info screen, and sliders for every parameter (with a copy-as-JSON button for new expressions):

```bash
open firmware/head_esp32/sim/index.html
```

**Previews as PNG files**, by the same renderer under node (no browser, no packages): each
expression at rest, with `--at` the model that many milliseconds after it was set (the breath, a
variant drawn, the heart in flight), with `--level` the mouth at those speech levels, and with
`--style` in that style:

```bash
node firmware/head_esp32/sim/preview.js --out /tmp/faces all
node firmware/head_esp32/sim/preview.js --out /tmp/faces --style anime --level 0.3,1 --gesture twitch,eps all
node firmware/head_esp32/sim/preview.js --out /tmp/faces --scale 3 --at 300,900 kiss_heart
```

## The serial protocol

Both ways: `0xA5 | type | length (u16 LE) | payload | CRC-8` (polynomial 0x07, initial 0, over
type, length and payload). `src/protocol.h` and `src/pepin/head_link.py` document every payload.

| type | direction | payload |
| --- | --- | --- |
| `I` | ESP32 -> host | IMU samples, n x 17 bytes: micros u32, ax ay az gx gy gz i16 (raw), config id u8 |
| `S` | ESP32 -> host | status once a second: fps, IMU rate, I2C errors, dropped samples, frame errors, ... |
| `P` | ESP32 -> host | pong: the ping's id, the ESP32's micros at receipt |
| `E` | host -> ESP32 | expression: id u8, intensity u8, transition ms u16 |
| `M` | host -> ESP32 | mouth openness u8 (0..255) |
| `T` | host -> ESP32 | info screen: duration ms u16, items (text, key/value, bar) |
| `Q` | host -> ESP32 | ping: id u32 |
| `C` | host -> ESP32 | config: id u8, IMU rate u16, DLPF u8, accel FS u8, gyro FS u8, brightness u8, face style u8 (`styles`' order: 0 pepin, 1 rose, 2 manga, 3 anime, 4 kawaii, 5 comics; a 7-byte frame leaves the style as it is) |

The IMU samples at the output rate the host's 'C' asks for (the head server asks 200 Hz,
`config/head.json`; the firmware boots at 1000 Hz until then), DLPF_CFG 3, +-4 g, +-500 deg/s,
CLKSEL 1 (the gyro's PLL), on its own core-0 task while the display runs on core 1. Each sample
is stamped with `micros()` in the data-ready interrupt, so the INT wire is required: without
edges the firmware streams nothing and reports `imu_state` 3 (a polled fallback beats against
the chip's own oscillator and is compiled out, `HEAD_IMU_POLL_FALLBACK`). Samples go out every
20 ms (4 at 200 Hz: 73 bytes, 4 % of the link; 20 at 1 kHz: 345 bytes, 19 %); a frame that does
not fit the serial buffer is dropped and counted, never waited for. The status line carries the
chip's rate measured in its own stamps (198-202 at a nominal 200 is the chip's oscillator, not a
fault) and the gaps longer than 1.5 periods.

## Built on

- [LovyanGFX](https://github.com/lovyan03/LovyanGFX) (panel driver, DMA, fonts; the info screen's
  IPA Gothic font covers Cyrillic)
- the Arduino core for ESP32 2.0.17 (PlatformIO espressif32 6.x)
