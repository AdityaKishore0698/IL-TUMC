"""Per-arm contrastive guidance plans for every ALMI-X gesture (play_trans.py --guidance_mode gesture).

Each arm is contrasted with a caption in which only that arm's role changes, as for waving:
  wave left        : left arm vs "wave right" (left rests),  right arm vs "wave both" (right moves)
  wave right       : left arm vs "wave both",                right arm vs "wave left"
  punch left       : left arm vs "punch right",              right arm vs "punch alternately with right and left hand"
  punch right      : left arm vs "punch alternately ...",    right arm vs "punch left"
  shake right hand : left arm vs "shake both hands",         right arm: no caption in which it rests ("shake left
                     hand" does not exist in ALMI-X)
  salute           : no contrast caption exists (ALMI-X has one salute caption per locomotion command)
  wave both, punch alternately, keep punching forward, shake both hands: not guided ("both" does not contradict
  either single-arm caption).
With fallback="loco", an arm without a contrast caption is contrasted with the locomotion-only caption (the
command without its gesture phrase). That caption never occurs in training, so this is exploratory.
"""
CONTRAST = {
    "wave left": ("wave right", "wave both"),
    "wave right": ("wave both", "wave left"),
    "punch left": ("punch right", "punch alternately with right and left hand"),
    "punch right": ("punch alternately with right and left hand", "punch left"),
    "shake right hand": ("shake both hands", None),
    "salute": (None, None),
}
SINGLE_ARM = {"wave left", "wave right", "punch left", "punch right", "shake right hand", "salute"}


def split_caption(text):
    """'Robot go forward slowly and punch left.' -> ('Robot go forward slowly', 'punch left')."""
    body = text.rstrip(".")
    loco, _, phrase = body.partition(" and ")
    return loco, phrase


def plan(text, fallback="none"):
    """-> (captions to run, [left-arm contrast index or None, right-arm contrast index or None])."""
    loco, phrase = split_caption(text)
    pair = CONTRAST.get(phrase)
    if pair is None:
        return [text], [None, None]
    texts, neg = [text], []
    for c in pair:
        if c is None and fallback == "loco" and phrase in SINGLE_ARM:
            caption = loco + "."
        elif c is None:
            neg.append(None)
            continue
        else:
            caption = f"{loco} and {c}."
        if caption not in texts:
            texts.append(caption)
        neg.append(texts.index(caption))
    return texts, neg
