"""Generate notes/waypoint_schemes_112_step.pdf from the 112-step walkthrough.

One-shot script — not part of the project. Lives in notes/ which is gitignored.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted, Table, TableStyle, HRFlowable,
)


OUT = "/home/kaboo/le-wm/notes/waypoint_schemes_112_step.pdf"

doc = SimpleDocTemplate(
    OUT, pagesize=letter,
    leftMargin=0.55 * inch, rightMargin=0.55 * inch,
    topMargin=0.55 * inch, bottomMargin=0.55 * inch,
    title="Waypoint schemes walkthrough (112-step episode)",
)

styles = getSampleStyleSheet()

title_st = ParagraphStyle('T', parent=styles['Title'], fontSize=15, spaceAfter=10, leading=18)
h2_st = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=12.5, spaceBefore=10, spaceAfter=5, textColor=HexColor('#1a1a3a'))
h3_st = ParagraphStyle('H3', parent=styles['Heading3'], fontSize=11, spaceBefore=8, spaceAfter=3, textColor=HexColor('#222'))
body_st = ParagraphStyle('B', parent=styles['BodyText'], fontSize=10, leading=13.5, spaceAfter=4)
code_st = ParagraphStyle('C', parent=styles['Code'], fontName='Courier', fontSize=8, leading=10,
                         leftIndent=8, backColor=HexColor('#f4f4f4'), borderPadding=4,
                         spaceBefore=3, spaceAfter=5, textColor=HexColor('#222'))


def hr():
    return HRFlowable(width="100%", thickness=0.5, color=HexColor('#bbb'),
                      spaceBefore=8, spaceAfter=8)


story = []

# ─── Title + intro ────────────────────────────────────────────────────────────
story.append(Paragraph("Walking each waypoint scheme through a 112-step episode", title_st))

story.append(Paragraph(
    "Imagine one TwoRoom-style trajectory with <b>112 raw env steps</b>, numbered 1 to 112:",
    body_st))
story.append(Preformatted(
    "1 2 3 4 5 6 7 8 9 10 ... 50 ... 100 ... 112\n"
    "+------------ 112 raw env steps -------------+",
    code_st))

story.append(Paragraph(
    'A "waypoint" is just a step number we pick to be a special anchor. Between two consecutive '
    'waypoints, the actions get packed into one <b>macro-action</b>. Each scheme picks waypoints '
    'differently.',
    body_st))

story.append(hr())

# ─── Scheme 1 — PLDM ──────────────────────────────────────────────────────────
story.append(Paragraph("Scheme 1 &mdash; PLDM (the closest analog to LeWM, Appendix B.4)", h2_st))
story.append(Paragraph(
    "<b>Rule:</b> chop out exactly 60 raw steps, then place waypoints every 10 steps. Deterministic.",
    body_st))
story.append(Preformatted(
    "Step 1: Pick a starting point, say step 1.\n"
    "Step 2: Cut out a window of 60 steps:\n"
    "        steps  1 -- 60                       (61..112 unused for this sample)\n"
    "Step 3: Place waypoints at fixed stride 10:\n"
    "        positions: 1, 11, 21, 31, 41, 51     <- 6 waypoints\n"
    "Step 4: Segments between waypoints:\n"
    "        [10, 10, 10, 10, 10]                 <- all equal, every time",
    code_st))
story.append(Paragraph("Visual:", body_st))
story.append(Preformatted(
    " 1    11   21   31   41   51       60                                  112\n"
    " W----W----W----W----W----W        |\n"
    " | 10 | 10 | 10 | 10 | 10 |\n"
    " +--- always identical segments ---+",
    code_st))
story.append(Paragraph(
    "Every training sample looks like this. <b>Zero within-sample variance.</b> "
    "Across samples, the only thing that changes is the starting point (1, 2, 3, ...) &mdash; "
    "segments are always 10 steps long.",
    body_st))

story.append(hr())

# ─── Scheme 2 — DINO-WM ───────────────────────────────────────────────────────
story.append(Paragraph("Scheme 2 &mdash; DINO-WM (Appendix B.3)", h2_st))
story.append(Paragraph(
    "<b>Rule:</b> randomize the <i>total span</i> per sample, then place 5 waypoints inside it.",
    body_st))
story.append(Preformatted(
    "Step 1: Sample segment length L ~ Uniform(25, 70). Say L = 50.\n"
    "Step 2: Pick a starting point. Say step 1. Window is steps 1..50.\n"
    "Step 3: Place 5 waypoints inside the 50-step window\n"
    "        (paper doesn't say how -- likely roughly uniform).\n"
    "        Example: positions 1, 13, 25, 37, 50.\n"
    "Step 4: Segments:\n"
    "        [12, 12, 12, 13]                     <- roughly equal, 1-step jitter",
    code_st))
story.append(Paragraph("Visual (one possible draw):", body_st))
story.append(Preformatted(
    " 1        13        25        37        50                              112\n"
    " W---------W---------W---------W---------W\n"
    " |   12    |   12    |   12    |   13    |",
    code_st))
story.append(Paragraph(
    "Across samples: span shifts (some L=27, some L=68), but within a single sample, the four "
    "segments stay similar. <b>Bounded within-sample variance.</b> "
    "The total span never grows past 70 or shrinks below 25.",
    body_st))

story.append(hr())

# ─── Scheme 3 — VJEPA2-AC ─────────────────────────────────────────────────────
story.append(Paragraph("Scheme 3 &mdash; VJEPA2-AC (Appendix B.2)", h2_st))
story.append(Paragraph(
    "<b>Rule:</b> sample a duration in seconds (0.33s to 4s), use only <b>3 waypoints</b> with the "
    "<b>middle one chosen uniformly at random</b>.",
    body_st))
story.append(Preformatted(
    "Step 1: Sample segment length. Say 30 steps.\n"
    "Step 2: Pick a starting point. Say step 1. Window is steps 1..30.\n"
    "Step 3: First and last waypoints are the endpoints.\n"
    "        Middle waypoint = uniform random somewhere inside.\n"
    "        Example: middle at step 12.\n"
    "        Positions: 1, 12, 30.\n"
    "Step 4: Segments: [11, 18]                   <- 2 segments only",
    code_st))
story.append(Paragraph("Visual:", body_st))
story.append(Preformatted(
    " 1            12                  30                                    112\n"
    " W-------------W-------------------W\n"
    " |     11      |        18         |",
    code_st))
story.append(Paragraph(
    "Only 2 segments per sample. Their ratio is whatever uniform-random luck draws &mdash; "
    "could be near-equal [14, 15] or lopsided [3, 26]. Bounded but noisy.",
    body_st))

story.append(hr())

# ─── Scheme 4 — Our current code ──────────────────────────────────────────────
story.append(Paragraph("Scheme 4 &mdash; Our current code", h2_st))
story.append(Paragraph(
    "<b>Rule:</b> fixed 100-step window after frameskip, but waypoints inside are placed by "
    "<font face='Courier'>torch.randperm</font> with no minimum gap.",
    body_st))
story.append(Preformatted(
    "Step 1: Cut out exactly 100 raw steps (span = frameskip*num_steps = 5*20).\n"
    "        Say starting at step 1. Window: 1..100.\n"
    "Step 2: The dataloader keeps every 5th step (frameskip). So our 'frames' are at\n"
    "        raw steps 1, 6, 11, 16, ..., 96. That's 20 frames, indexed 0..19.\n"
    "Step 3: sample_waypoints(T=20, N=3) returns N+2 = 5 frame indices.\n"
    "        Always includes frame 0 and frame 19. Picks 3 more uniformly at random.",
    code_st))
story.append(Paragraph(
    "Now the question is <b>which 3 interior frames get picked.</b> Here are two possible draws:",
    body_st))

story.append(Paragraph("Draw A (typical)", h3_st))
story.append(Preformatted(
    "frame indices: [0, 4, 11, 16, 19]\n"
    "raw steps:     [1, 21, 56, 81, 96]\n"
    "segments (raw steps): [20, 35, 25, 15]",
    code_st))
story.append(Preformatted(
    " 1          21                       56                  81           96    100        112\n"
    " W-----------W------------------------W--------------------W-------------W\n"
    " |     20    |           35           |         25         |      15     |",
    code_st))
story.append(Paragraph(
    "Segment ratio 35:15 = <b>2.3x</b>. A bit uneven but tolerable.",
    body_st))

story.append(Paragraph("Draw B (pathological &mdash; also legal)", h3_st))
story.append(Preformatted(
    "frame indices: [0, 1, 2, 3, 19]\n"
    "raw steps:     [1, 6, 11, 16, 96]\n"
    "segments (raw steps): [5, 5, 5, 80]",
    code_st))
story.append(Preformatted(
    " 1 6 11 16                                                   96      100       112\n"
    " WWWWW-------------------------------------------------------W\n"
    " |5|5|5|                           80                        |",
    code_st))
story.append(Paragraph(
    "Segment ratio 80:5 = <b>16x</b>. A_psi is asked to compress <i>one effective action</i> "
    "(5 raw steps) into a macro-action for the first three segments, then compress <i>sixteen "
    "effective actions</i> (80 raw steps) into a macro-action for the last one &mdash; all in "
    "the same training example.",
    body_st))

story.append(hr())

# ─── Side-by-side comparison ──────────────────────────────────────────────────
story.append(Paragraph("Side-by-side on the 112-step episode", h2_st))

table_data = [
    ['Scheme', 'Window', '# waypoints', '# segments', 'Example segments', 'Max ratio in one sample'],
    ['PLDM',       'fixed 60',      '6', '5', '[10, 10, 10, 10, 10]',      '1x (always equal)'],
    ['DINO-WM',    'Unif(25, 70)',  '5', '4', '[12, 12, 12, 13]',          '~1.1x'],
    ['VJEPA2-AC',  'Unif(3, 40)',   '3', '2', '[11, 18]',                  'a few x'],
    ['Our code',   'fixed 100',     '5', '4', '[20, 35, 25, 15] typical',  'typ ~2.5x, worst 16x'],
]

t = Table(table_data, colWidths=[0.8 * inch, 0.95 * inch, 0.8 * inch, 0.8 * inch, 1.7 * inch, 1.5 * inch], repeatRows=1)
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e8e8f4')),
    ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE',   (0, 0), (-1, -1), 8.5),
    ('GRID',       (0, 0), (-1, -1), 0.4, HexColor('#888')),
    ('VALIGN',     (0, 0), (-1, -1), 'TOP'),
    ('LEFTPADDING',  (0, 0), (-1, -1), 4),
    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ('TOPPADDING',   (0, 0), (-1, -1), 3),
    ('BOTTOMPADDING',(0, 0), (-1, -1), 3),
    # Highlight "Our code" row
    ('BACKGROUND', (0, 4), (-1, 4), HexColor('#fff4e0')),
    ('FONTNAME',   (0, 4), (0, 4), 'Helvetica-Bold'),
    ('FONTNAME',   (0, 1), (0, 1), 'Helvetica-Bold'),
]))
story.append(t)

story.append(hr())

# ─── The point ────────────────────────────────────────────────────────────────
story.append(Paragraph("The point", h2_st))
story.append(Paragraph(
    "PLDM &mdash; the JEPA-from-pixels backbone, which is the design closest to LeWM &mdash; "
    "picks waypoints <b>at fixed stride 10</b>, every time. No randomness inside the sample. "
    "Our code's \"random interior frames, no minimum gap\" gives the model wildly inconsistent "
    "training signals about what one macro-action represents (sometimes 5 raw steps, sometimes 80). "
    "That's the concern.",
    body_st))
story.append(Paragraph(
    "<b>Fix:</b> replace <font face='Courier'>torch.randperm</font> with a fixed stride. For our "
    "T=20, N=3 setup, that means picking frame indices <b>[0, 5, 10, 15, 19]</b> (or "
    "[0, 4, 9, 14, 19]) every sample &mdash; same spec PLDM used in HWM.",
    body_st))

doc.build(story)
print(f"wrote {OUT}")
