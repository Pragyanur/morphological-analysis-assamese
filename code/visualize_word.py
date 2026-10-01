"""
Word -> morphological analyses -> image of each analysis's morphological vector.

    python code/visualize_word.py ককাকে
    python code/visualize_word.py কৈছিল -o kaisil.png
    python code/visualize_word.py মানুহবোৰক --font /path/to/NotoSansBengali-Regular.ttf

For every analysis the rule-based analyser returns, the word is vectorised with
`vectorize_morphology` (the 64-d vector used by the disambiguation model) and only
the non-zero features are drawn: one panel per analysis, one bar per feature,
labelled with its name, its index in the 64-d vector, and its value.

Assamese text is shaped with Pillow + libraqm (matplotlib does not shape Indic
scripts, so vowel signs and conjuncts would otherwise render out of order).
A Bengali-script font is needed for the Assamese labels; pass one with --font or
let the script search common system locations.
"""

import argparse
import glob
import os
import sys
import unicodedata

# morphology.py opens "resources/..." relative to the working directory, so
# import it from the repo root regardless of where this script is run from.
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CODE_DIR)
_cwd = os.getcwd()
os.chdir(REPO_ROOT)
sys.path.insert(0, CODE_DIR)
from morphology import MORPHOLOGY_FEATURES, sentence_word_options, vectorize_morphology  # noqa: E402
os.chdir(_cwd)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# vectorize_morphology builds the vector from sorted(vector_dict.items()),
# so index i of the vector is the i-th feature name in sorted order.
FEATURE_NAMES = sorted(MORPHOLOGY_FEATURES)

# ---- feature groups (colour = group, so the bars read by kind) -------------
POS_FEATURES = {
    "Abstract Noun", "Adjective", "Adjective Adj.", "Adposition", "Adverb",
    "Common Noun", "Conjuction", "Conjunction", "Interjection", "Material Noun",
    "Noun", "Others", "Pronoun", "Proper Adj.", "Proper Noun", "Verb",
    "Verb-Intran.", "Verb-Trans.", "Verbable", "Verbal Adj.", "Verbal Noun",
    "numeric-string", "numerical",
}
CASE_FEATURES = {"abs", "acc", "dat", "erg", "gen", "inst", "loc", "t_loc"}
AGREEMENT_FEATURES = {"p1", "p2", "p3", "a", "i", "n", "f", "past", "present", "future"}
NUMBER_FEATURES = {"number", "sg", "pl"}

GROUPS = [  # (label, colour) - fixed order, default categorical palette
    ("POS / root", "#2a78d6"),
    ("case", "#eb6834"),
    ("tense / person / honorific", "#1baf7a"),
    ("number", "#eda100"),
    ("other suffix / symbol", "#e87ba4"),
]
TEXT_PRIMARY, TEXT_SECONDARY, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def feature_group(name):
    if name in POS_FEATURES:
        return 0
    if name in CASE_FEATURES:
        return 1
    if name in AGREEMENT_FEATURES:
        return 2
    if name in NUMBER_FEATURES:
        return 3
    return 4


# ---- Assamese text shaping -------------------------------------------------
FONT_SEARCH = [
    "~/fonts/*Bengali*.ttf",
    "/usr/share/fonts/**/*Bengali*.ttf",
    "/usr/share/fonts/**/*Assamese*.ttf",
    "/usr/share/fonts/**/Lohit-Assamese*.ttf",
    "/usr/share/fonts/**/Lohit-Bengali*.ttf",
    "/usr/share/fonts/**/Mukti*.ttf",
    "/Library/Fonts/*Bangla*.tt*",
    "/System/Library/Fonts/**/*Bangla*.tt*",
    "C:/Windows/Fonts/Nirmala*.ttf",
    "C:/Windows/Fonts/Vrinda*.ttf",
]


def find_font(user_font=None):
    if user_font:
        return user_font
    for pattern in FONT_SEARCH:
        hits = glob.glob(os.path.expanduser(pattern), recursive=True)
        if hits:
            return sorted(hits)[0]
    return None


class Shaper:
    """Renders text to an RGBA array with correct Indic shaping (Pillow + raqm)."""

    def __init__(self, font_path, dpi):
        self.ok = False
        self.font_path = font_path
        self.dpi = dpi
        if not font_path:
            return
        try:
            from PIL import ImageFont, features
            if not features.check("raqm"):
                print("warning: Pillow has no libraqm; Assamese labels may render "
                      "with broken shaping.", file=sys.stderr)
                return
            self.ImageFont = ImageFont
            self.ok = True
        except ImportError:
            return

    @staticmethod
    def _runs(text):
        """Split text into (is_bengali_script, substring) runs."""
        runs = []
        for ch in text:
            beng = "ঀ" <= ch <= "৿" or ch in "‌‍"
            if runs and runs[-1][0] == beng:
                runs[-1][1] += ch
            else:
                runs.append([beng, ch])
        return runs

    def image(self, text, pt=11, color=TEXT_PRIMARY):
        """Assamese runs in the Bengali-script font, everything else in
        matplotlib's own font, laid out on a shared baseline."""
        from PIL import Image, ImageDraw
        from matplotlib import font_manager
        px = int(round(pt * self.dpi / 72))
        raqm = self.ImageFont.Layout.RAQM
        fonts = {
            True: self.ImageFont.truetype(self.font_path, px, layout_engine=raqm),
            False: self.ImageFont.truetype(font_manager.findfont("DejaVu Sans"), px,
                                           layout_engine=raqm),
        }
        runs = self._runs(text)
        ascent = max(f.getmetrics()[0] for f in fonts.values())
        descent = max(f.getmetrics()[1] for f in fonts.values())
        widths = [fonts[b].getlength(s, language="as" if b else None) for b, s in runs]
        pad = max(2, px // 6)
        img = Image.new("RGBA", (int(sum(widths)) + 2 * pad, ascent + descent + 2 * pad),
                        (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        x = pad
        for (b, s), w in zip(runs, widths):
            draw.text((x, pad + ascent), s, font=fonts[b], fill=color, anchor="ls",
                      language="as" if b else None)
            x += w
        return np.asarray(img)


def place_text(ax, shaper, text, xy, xycoords, pt=11, color=TEXT_PRIMARY,
               ha=0.0, va=0.5):
    """Put (possibly Assamese) text on the axes; box_alignment (ha, va) in 0..1."""
    if shaper.ok:
        # OffsetImage zoom is in points per pixel; the array is rendered at figure dpi
        box = OffsetImage(shaper.image(text, pt, color), zoom=72.0 / shaper.dpi)
        ab = AnnotationBbox(box, xy, xycoords=xycoords, frameon=False,
                            box_alignment=(ha, va), pad=0, annotation_clip=False)
        ax.add_artist(ab)
    else:
        ax.annotate(text, xy, xycoords=xycoords, fontsize=pt, color=color,
                    ha={0.0: "left", 0.5: "center", 1.0: "right"}[ha],
                    va={0.0: "bottom", 0.5: "center", 1.0: "top"}[va],
                    annotation_clip=False)


# ---- analysis --------------------------------------------------------------
def analyse(word):
    """All analyses for `word`, with their 64-d vectors."""
    options = sentence_word_options(word)[0]
    return [(seq, vectorize_morphology(seq)) for seq in options]


def describe(seq):
    """'ককা/Common Noun + ক/acc + ে/emph' (empty morphemes = inferred tags)."""
    return "  +  ".join(f"{m}/{t}" if m else f"({t})" for m, t in seq)


# ---- drawing ---------------------------------------------------------------
def draw(word, analyses, out_path, shaper, dpi):
    n = len(analyses)
    heights = [max(1, int(np.count_nonzero(v))) for _, v in analyses]
    row_in = 0.34
    panel_heights = [h * row_in + 0.95 for h in heights]
    fig_h = sum(panel_heights) + 1.15
    fig, axes = plt.subplots(n, 1, figsize=(8.2, fig_h), dpi=dpi,
                             gridspec_kw={"height_ratios": panel_heights}, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    vmax = max(1.0, max(float(v.max()) for _, v in analyses))

    for k, (ax, (seq, vec)) in enumerate(zip(axes[:, 0], analyses)):
        ax.set_facecolor(SURFACE)
        idx = np.flatnonzero(vec)
        names = [FEATURE_NAMES[i] for i in idx]
        vals = vec[idx]
        ys = np.arange(len(idx))[::-1]
        colours = [GROUPS[feature_group(nm)][1] for nm in names]
        ax.barh(ys, vals, height=0.62, color=colours, edgecolor=SURFACE, linewidth=2, zorder=3)

        for y, nm, i, v in zip(ys, names, idx, vals):
            ax.text(-0.03 * vmax, y, f"{nm}  [{i}]", ha="right", va="center",
                    fontsize=9.5, color=TEXT_PRIMARY)
            ax.text(v + 0.02 * vmax, y, f"{v:g}", ha="left", va="center",
                    fontsize=9, color=TEXT_SECONDARY)

        ax.set_xlim(0, vmax * 1.12)
        ax.set_ylim(-0.6, max(len(idx), 1) - 0.4)
        ax.set_yticks([])
        ax.tick_params(axis="x", colors=TEXT_SECONDARY, labelsize=8, length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

        place_text(ax, shaper, f"analysis {k + 1}/{n}:   {describe(seq)}",
                   (0.03, 1.0), ("figure fraction", "axes fraction"), pt=10.5, color=TEXT_PRIMARY, ha=0.0, va=0.0)
        if len(idx) == 0:
            ax.text(0.5, 0.5, "all-zero vector", transform=ax.transAxes,
                    ha="center", va="center", color=TEXT_SECONDARY)

    # title (the word itself) and legend for the groups actually used
    title_ax = fig.add_axes([0, 0, 1, 1], facecolor="none")
    title_ax.set_axis_off()
    place_text(title_ax, shaper, word, (0.03, 1 - 0.18 / fig_h), "axes fraction",
               pt=18, color=TEXT_PRIMARY, ha=0.0, va=1.0)
    title_ax.text(0.97, 1 - 0.30 / fig_h,
                  f"{n} analys{'is' if n == 1 else 'es'} · non-zero features of the 64-d morphological vector",
                  ha="right", va="center", fontsize=9, color=TEXT_SECONDARY)
    used = sorted({feature_group(FEATURE_NAMES[i]) for _, v in analyses for i in np.flatnonzero(v)})
    handles = [Patch(color=GROUPS[g][1], label=GROUPS[g][0]) for g in used]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=8.5, labelcolor=TEXT_SECONDARY, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.0, handleheight=0.8)

    fig.subplots_adjust(left=0.30, right=0.95, top=1 - 0.95 / fig_h,
                        bottom=0.55 / fig_h, hspace=0.95 / (sum(panel_heights) / n))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("word", help="an Assamese word (or symbol / number)")
    p.add_argument("-o", "--output", help="image path (.png, .pdf, .svg); default <word>_morphology.png")
    p.add_argument("--font", help="Bengali-script .ttf/.otf for the Assamese labels")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    word = unicodedata.normalize("NFC", args.word.strip())
    if not word or len(word.split()) != 1:
        p.error("give exactly one word")

    analyses = analyse(word)

    print(f"{word}: {len(analyses)} analysis(es)")
    for k, (seq, vec) in enumerate(analyses, 1):
        nz = ", ".join(f"{FEATURE_NAMES[i]}[{i}]={vec[i]:g}" for i in np.flatnonzero(vec))
        print(f"  {k}. {describe(seq)}\n     {nz}")

    font = find_font(args.font)
    if not font:
        print("warning: no Bengali-script font found; pass --font. Assamese labels "
              "will render as boxes.", file=sys.stderr)
    shaper = Shaper(font, args.dpi)

    out = args.output or f"{word}_morphology.png"
    draw(word, analyses, out, shaper, args.dpi)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
