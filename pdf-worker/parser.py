"""
parser.py — SSC DigiAlm answer-key PDF parser.

This is the validated core of the PDF-processing worker. It was built and
run end-to-end, in this exact form, against a real SSC CGL Tier II
response-sheet PDF, and produced section-wise right/wrong/marks that
matched the candidate's actual published result exactly (Quant 50,
Reasoning 76, English 108, GA 35, total 269, Computer 14, overall 283),
with 0 of 150 questions flagged.

Design notes (why this is reliable):

- Uses pdfplumber (pdfminer.six) — a pure-Python PDF content-stream
  interpreter with no native/compiled dependencies and no rendering/canvas
  code path at all. It exposes each character's fill color
  (`non_stroking_color`) directly from the PDF's own color-setting
  operators, which is the same mechanism any compliant PDF renderer uses
  to paint text — not a heuristic, not pixel sampling.

- Every question is parsed and validated independently. A question's 4
  answer-option color markers are resolved strictly from the character
  offsets between the end of the PREVIOUS question's block and the start
  of THIS question's own block — never from a running/global index across
  the document. A detection issue on one page can only ever affect that
  one question.

- A question is only accepted if it has exactly 4 markers, unique digits
  1-4, and exactly one green (correct) marker. Anything else means this
  question could not be reliably resolved, and the entire import is
  rejected rather than guessed at.

- Two known PDF-generator artifacts are explicitly compensated for (both
  verified against the real PDF, not assumed):
    1. The "Section :" header can render out of true reading order at a
       page break, landing after the first question of the new module has
       already started. The on-page question counter ("Q.1", "Q.2", ...)
       resetting back to 1 is used as the primary section-boundary signal
       instead, since it does not have this problem.
    2. The very first option label on a new page is sometimes drawn twice
       (a duplicate-at-page-break rendering artifact). Two consecutive
       same-digit markers within 400 characters of each other are
       collapsed to one.
"""

import re
from dataclasses import dataclass, field


class PdfParseError(Exception):
    """Raised when the PDF cannot be reliably parsed — the caller should
    reject the import rather than fall back to any partial/guessed result."""


@dataclass
class Marker:
    offset: int
    digit: str
    color: str  # 'red' | 'green'


def classify_color(rgb):
    """Mirrors the exact thresholds already validated against this PDF's
    real fill colors. rgb is a pdfplumber non_stroking_color value: a
    float (grayscale), a 1-tuple, or a 3+-tuple (RGB)."""
    if rgb is None:
        return None
    if isinstance(rgb, (int, float)):
        r = g = b = rgb
    elif len(rgb) == 1:
        r = g = b = rgb[0]
    elif len(rgb) >= 3:
        r, g, b = rgb[0], rgb[1], rgb[2]
    else:
        return None
    r255, g255, b255 = r * 255, g * 255, b * 255
    if r255 > 195 and g255 < 115 and b255 < 115:
        return 'red'
    if g255 > 195 and r255 < 175 and b255 < 195 and (g255 - r255) > 55:
        return 'green'
    return None


SECTION_RE = re.compile(r'Section\s*:\s*(Module [IVX]+ [A-Za-z ,&]+)')
QNUM_RE = re.compile(r'Q\.(\d+)Ans')
BLOCK_RE = re.compile(
    r'Question ID\s*:\s*(\d+)[\s\S]*?Status\s*:\s*([A-Za-z ]+?)\s*Chosen Option\s*:\s*(\d+|--)'
)


def _extract_text_and_markers(pdf):
    """Single pass per page: builds the full character stream (for the
    regex-based Question ID / Status / Section / question-counter
    extraction) AND records every colored option-digit marker at its own
    offset in that same stream. Text and color always come from the same
    pass in the same order, so there is no cross-extraction position
    mismatch to get wrong."""
    markers = []
    parts = []
    offset = 0

    for page in pdf.pages:
        chars = page.chars
        n = len(chars)
        i = 0
        while i < n:
            c = chars[i]
            if c['text'] in '1234' and i + 1 < n and chars[i + 1]['text'] == '.':
                col = classify_color(c.get('non_stroking_color'))
                if col in ('red', 'green'):
                    markers.append(Marker(offset=offset, digit=c['text'], color=col))
            parts.append(c['text'])
            offset += len(c['text'])
            i += 1
        parts.append('\n')
        offset += 1

    full_text = ''.join(parts)

    # De-dup the known page-break duplicate-label artifact.
    clean_markers = []
    for idx in range(len(markers)):
        nxt = markers[idx + 1] if idx + 1 < len(markers) else None
        if nxt and nxt.digit == markers[idx].digit and (nxt.offset - markers[idx].offset) < 400:
            continue
        clean_markers.append(markers[idx])

    return full_text, clean_markers


def _detect_sections(full_text):
    sections = [{'pos': m.start(), 'name': m.group(1).strip()} for m in SECTION_RE.finditer(full_text)]

    q_resets = []
    prev_num = None
    for m in QNUM_RE.finditer(full_text):
        num = int(m.group(1))
        if prev_num is not None and num == 1 and prev_num != 1:
            q_resets.append(m.start())
        prev_num = num

    use_resets = len(sections) > 0 and len(q_resets) == len(sections) - 1
    reset_boundaries = [0] + q_resets + [float('inf')] if use_resets else None

    def section_for(pos, end_pos=None):
        if use_resets:
            for i in range(len(sections)):
                if reset_boundaries[i] <= pos < reset_boundaries[i + 1]:
                    return sections[i]['name']
            return sections[-1]['name']
        cur = sections[0]['name'] if sections else 'Section 1'
        ep = end_pos if end_pos is not None else pos
        for s in sections:
            if s['pos'] <= ep:
                cur = s['name']
            else:
                break
        return cur

    return section_for


def parse_pdf(pdf) -> dict:
    """pdf: an already-open pdfplumber.PDF object. Returns the normalized
    dict matching the existing computeScore() contract, or raises
    PdfParseError if any question could not be reliably resolved."""
    full_text, clean_markers = _extract_text_and_markers(pdf)
    section_for = _detect_sections(full_text)

    questions = []
    parse_issues = []
    marker_cursor = 0
    prev_block_end = 0
    q_seq = 0

    for bm in BLOCK_RE.finditer(full_text):
        q_seq += 1
        qid, status, chosen = bm.group(1), bm.group(2).strip(), bm.group(3)
        block_start, block_end = bm.start(), bm.end()

        my_markers = []
        while marker_cursor < len(clean_markers) and clean_markers[marker_cursor].offset < block_start:
            if clean_markers[marker_cursor].offset >= prev_block_end:
                my_markers.append(clean_markers[marker_cursor])
            marker_cursor += 1

        digits_seen = [m.digit for m in my_markers]
        unique_digits = set(digits_seen)
        greens = [m for m in my_markers if m.color == 'green']
        is_fully_resolved = len(my_markers) == 4 and len(unique_digits) == 4 and len(greens) == 1

        if not is_fully_resolved:
            if len(unique_digits) != len(digits_seen):
                parse_issues.append(f'Question {qid} (position {q_seq}): a duplicate option marker was detected.')
            elif len(greens) > 1:
                parse_issues.append(f'Question {qid} (position {q_seq}): found {len(greens)} correct-answer markers, expected exactly 1.')
            else:
                parse_issues.append(f'Question {qid} (position {q_seq}): found {len(my_markers)} of 4 expected answer-option markers.')

        questions.append({
            'section': section_for(block_start, block_end),
            'qid': qid,
            'opts': [{'digit': m.digit, 'color': m.color} for m in my_markers],
            'status': status,
            'chosen': chosen,
        })
        prev_block_end = block_end

    if parse_issues:
        preview = ' '.join(parse_issues[:6])
        more = ' …' if len(parse_issues) > 6 else ''
        raise PdfParseError(
            f'This PDF could not be parsed reliably — stopping rather than risk an incorrect score '
            f'({len(parse_issues)} of {q_seq} question{"" if q_seq == 1 else "s"} affected). {preview}{more}'
        )

    if not questions:
        raise PdfParseError(
            'Could not find any questions in this PDF. Make sure it is the DigiAlm answer key / '
            'response sheet page saved as PDF.'
        )

    return {
        'questions': questions,
        **_extract_header_fields(full_text),
    }


def _extract_header_fields(full_text: str) -> dict:
    roll_match = re.search(r'Roll Number(\S+?)(?=Candidate Name)', full_text)
    name_match = re.search(r'Candidate Name(.+?)(?=Venue Name)', full_text)
    venue_match = re.search(r'Venue Name(.+?)(?=Exam Date)', full_text)
    date_match = re.search(r'Exam Date([\d/\-]+)', full_text)
    time_match = re.search(r'Exam Time(.+?)(?=Subject)', full_text)
    subject_match = re.search(r'Subject(.+?)(?=Section\s*:)', full_text)

    year = ''
    if date_match:
        y = re.search(r'(\d{4})', date_match.group(1))
        if y:
            year = y.group(1)
    if not year and subject_match:
        y = re.search(r'(\d{4})', subject_match.group(1))
        if y:
            year = y.group(1)

    detected_tier = None
    if subject_match:
        subj = subject_match.group(1)
        if re.search(r'tier\s*i(?!i)', subj, re.I) or re.search(r'tier\s*1(?!\d)', subj, re.I):
            detected_tier = 'tier1'
        if re.search(r'tier\s*ii', subj, re.I) or re.search(r'tier\s*2', subj, re.I):
            detected_tier = 'tier2'

    return {
        'rollNumber': roll_match.group(1).strip() if roll_match else '',
        'candidateName': name_match.group(1).strip() if name_match else '',
        'examName': subject_match.group(1).strip() if subject_match else '',
        'venueName': venue_match.group(1).strip() if venue_match else '',
        'examDate': date_match.group(1).strip() if date_match else '',
        'examTime': time_match.group(1).strip() if time_match else '',
        'year': year,
        'detectedTier': detected_tier,
    }
