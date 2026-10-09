# AISTATS 2027 figure and table review

Reviewed on 1 October 2026 against the official
[2027 paper pack](https://aistats.org/aistats2027/AISTATS2027PaperPack.zip)
and [call for papers](https://virtual.aistats.org/Conferences/2027/CallForPapers).

The separate Overleaf package's `aistats2027.sty` and `fancyhdr.sty` match the official download
byte for byte. The submission uses the anonymous review option, US Letter,
the template's text area and columns, and eight pages of main content.
The 16-page supplement uses the same style in one column. The complete PDF
has 27 pages. Body fonts and official margins are unchanged.

## Figures and tables

- All six figures and ten tables are numbered consecutively and referenced
  in the text. There are eight empirical tables, one conceptual comparison
  and one protocol table describing data roles. The earlier study overview is
  retained in the artifact documentation.
- Figure captions are below the graphics and remain on the same page.
  The space before a figure caption is two 11-point lines.
- Table captions are above the tables, with one 11-point line before and after
  the caption. Tables are centered and their rules do not touch caption text.
- Captions identify the relevant conditions, aggregation and interpretation of
  uncertainty where it is displayed. Calibration ratios target one. Quality
  metrics state the better direction. Disagreement is distinguished from quality.
- Panel headings name corpora or conditions. Legends are within the figure
  boundary, outside data where practical. Row-specific dotted-line meanings are
  stated in the figure or caption. The template does not prescribe a universal
  inside-versus-outside legend position.
- Axes identify quantities, chunk counts, percentages, normalization and log
  scales where used. Connecting lines and missing confidence intervals are
  described where needed.
- All plots are vector PDFs with embedded fonts. The revised paper contains no
  Type 3 fonts. Top-k labels were enlarged and its repeated internal title was
  removed. The inverse-size plot now has larger labels and leaves the repeated
  experimental details to its caption.
- The inverse-size figure and the calibration figure both check that their
  y-axis labels have at least eight pixels of clearance inside the figure.
  The historical calibration figure uses two marker categories: existing A/B outcomes and forecasts
  evaluated on fresh A/B outcomes. All 186 plotted points and their intervals
  are preserved. The appendix gives the detailed study chronology. The new main
  calibration figure uses separate colors and markers for four methods, with
  crossed bank/pair intervals and a distinct larger-reference benchmark. The
  corrected multinomial legend uses its method name, and captions point to
  the separate prospective confirmation. The experimental sequence is documented
  in the protocol text.

The official template requires legible graphics. It does not state a numerical
minimum font size for plot labels. Font embedding, redundant line/marker cues,
and avoiding repeated titles are presentation choices, not additional claimed
conference rules.

## Other formatting corrections

The abstract remains one paragraph, as required by the template. The prose was
revised throughout the main paper and appendices for connected explanations
and sentence variety. Section 4 now follows the scientific questions, with
the corrected-reference comparison leading the main empirical evidence. A compact
data-role table appears in Section 2, and a sparse-row smoothing example
introduces Section 3. Margins, body font, line spacing and official style files
are unchanged.

This review checks the rendered document and package consistency. It does not
replace the authors' final submission checks or guarantee an editorial decision.
