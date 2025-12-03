
"""
Heuristic event miner from MOT tracks (single results root, both classes):
- pass: A -> None -> B with enough air time AND enough ball travel.
- dribble: same owner with periodic near/far toggles (>=2 cycles).
- shot: owner -> None sustained air; stronger gate with speed near release.
Writes JSON per sequence: results/actions_auto/<split>/<seq>.json
"""
import argparse, json, math, re
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np 

# ------------------ defaults (scale to image width) ------------------
FPS_DEFAULT = 2
# distance thresholds are referenced at width=1280 and scaled by W/1280
OWN_MAX_PX      = 120     # owner acquire distance
OWN_HYSTERESIS  = 20      # stickiness margin
NEAR_PX         = 40      # dribble 'near'
FAR_PX          = 90      # dribble 'far'
SHOT_MIN_AIR_F  = 10      # frames of None after release
SHOT_MIN_SPEED  = 8.0     # px/frame around release

# NEW: quality gates (still scaled by W)
PASS_MIN_AIR_F      = 3               # must have at least this many None frames between owners
PASS_MAX_AIR_F      = 45
PASS_MIN_TRAVEL_PX  = 80              # ball release->catch center travel (scaled)
DEBOUNCE_F          = 4               # suppress duplicate events within ±k frames
MIN_HOLD_F          = 2               # owner must hold at least k frames to be actionable
BALL_SMOOTH_F       = 1               # moving-average radius for ball center smoothing

# --------------------------------------------------------------------
def read_seqinfo(seq_dir: Path):
    W = H = L = None
    fps = FPS_DEFAULT
    f = seq_dir / "seqinfo.ini"
    if f.exists():
        for line in f.read_text().splitlines():
            if line.startswith("imWidth="):   W = int(line.split("=",1)[1])
            if line.startswith("imHeight="):  H = int(line.split("=",1)[1])
            if line.startswith("seqLength="): L = int(line.split("=",1)[1])
            if line.startswith("frameRate="): fps = int(line.split("=",1)[1])
    return W, H, L, fps

def load_mot(mot_file: Path):
    """Return dict frame->list of (tid, cls, x,y,w,h, conf)."""
    by_f = defaultdict(list)
    if not mot_file.exists():
        return by_f
    for line in mot_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        toks = re.split(r"[,\s]+", line)
        if len(toks) < 8:
            continue
        fr   = int(float(toks[0])); tid = int(float(toks[1]))
        x,y,w,h = map(float, toks[2:6])
        conf = float(toks[6])
        cls  = int(float(toks[7]))
        by_f[fr].append((tid, cls, x,y,w,h, conf))
    return by_f

def cxcy_from_box(box):
    _,_,x,y,w,h,_ = box
    return (x + w*0.5, y + h*0.5)

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def scale_by_width(W, v):
    if not W:
        return v
    return v * max(0.6, min(1.6, W/1280.0))

def velocity(series):
    return [0.0] + [dist(series[i], series[i-1]) for i in range(1, len(series))]

# ----------------------- smoothing helpers --------------------------
def smooth_centers(seq_frames, centers_by_f, radius=1):
    if radius <= 0 or not centers_by_f:
        return dict(centers_by_f)
    frames = [f for f in seq_frames if f in centers_by_f]
    out = {}
    for i, f in enumerate(frames):
        lo = max(0, i - radius); hi = min(len(frames)-1, i + radius)
        xs, ys, n = 0.0, 0.0, 0
        for j in range(lo, hi+1):
            x,y = centers_by_f[frames[j]]
            xs += x; ys += y; n += 1
        out[f] = (xs/n, ys/n)
    return out

def majority_smooth_owners(owner_series, radius=1):
    if radius <= 0 or not owner_series:
        return owner_series
    frames = [f for f,_ in owner_series]
    vals   = [o for _,o in owner_series]
    out = []
    for i in range(len(vals)):
        lo = max(0, i - radius); hi = min(len(vals)-1, i + radius)
        window = vals[lo:hi+1]
        counts = Counter([v for v in window if v is not None])
        if counts:
            m = counts.most_common(1)[0][0]
            out.append((frames[i], m))
        else:
            out.append((frames[i], vals[i]))
    return out

# ----------------------- ownership inference ------------------------
def compute_ownership(frames, players_by_f, ball_by_f, own_max, own_hys, player_cls, ball_cls, min_hold_f=MIN_HOLD_F):
    """
    Per-frame owner heuristic with hysteresis and min-hold.
    FIX 1: Calculates distance to last_owner in CURRENT frame to prevent teleporting.
    """
    owner_raw = {}
    last_owner = None
    hold_len   = 0

    for f in frames:
        # select best ball
        balls = [b for b in ball_by_f.get(f, []) if b[1]==ball_cls]
        if not balls:
            owner_raw[f] = None
            last_owner = None; hold_len = 0
            continue
        ball = max(balls, key=lambda z:z[-1])
        bc = cxcy_from_box(ball)

        # find nearest player
        players = [p for p in players_by_f.get(f, []) if p[1]==player_cls]
        if not players:
            owner_raw[f] = None
            last_owner = None; hold_len = 0
            continue

        min_d, min_id = 1e9, None
        last_owner_curr_d = 1e9 # Distance to last_owner in THIS frame

        for p in players:
            d = dist(bc, cxcy_from_box(p))
            if d < min_d:
                min_d, min_id = d, p[0]
            if last_owner is not None and p[0] == last_owner:
                last_owner_curr_d = d

        # hysteresis: use current frame distance
        if last_owner is not None and min_id != last_owner:
            # If we are still within reasonable range of last_owner, stick to them
            if last_owner_curr_d <= own_max:
                min_id = last_owner
                min_d = last_owner_curr_d

        if min_d <= own_max:
            if min_id == last_owner:
                hold_len += 1
            else:
                hold_len = 1
            last_owner = min_id
            owner_raw[f] = min_id if hold_len >= min_hold_f else None
        else:
            owner_raw[f] = None
            last_owner = None; hold_len = 0

    return majority_smooth_owners([(f, owner_raw.get(f)) for f in frames], radius=1)

# ----------------------- primitive kinematics -----------------------
def velocity_series(frames, centers):
    v = {frames[0]: 0.0}
    for i in range(1, len(frames)):
        f0, f1 = frames[i-1], frames[i]
        if (f0 in centers) and (f1 in centers):
            v[f1] = dist(centers[f1], centers[f0])
        else:
            v[f1] = 0.0
    return v

# ----------------------- event detectors ---------------------------
def detect_pass(owner_series, ball_centers, near_thr, min_air_f, max_air_f, min_travel_px, debounce_f):
    """
    FIX 2: Searches for ball ±1 frame to handle occlusion at catch/release.
    FIX 3: Actually checks 'near_thr'.
    """
    events = []
    last_emit = -10**9

    def get_ball_loc(target_f):
        # Look 1 frame back/forward if exact frame missing
        for off in [0, -1, 1]:
            if (target_f + off) in ball_centers: return ball_centers[target_f+off]
        return None

    # make owner runs
    i = 0; n = len(owner_series)
    while i < n:
        f0, oid = owner_series[i]
        j = i
        while j+1 < n and owner_series[j+1][1] == oid:
            j += 1
        if oid is not None:
            release_f = owner_series[j][0]
            k = j + 1
            air = 0
            while k < n and owner_series[k][1] is None:
                air += 1; k += 1
            
            if k < n and owner_series[k][1] is not None and min_air_f <= air <= max_air_f:
                catch_f, new_owner = owner_series[k][0], owner_series[k][1]
                if new_owner != oid:
                    b_rel = get_ball_loc(release_f)
                    b_cat = get_ball_loc(catch_f)
                    
                    if b_rel and b_cat:
                        travel = dist(b_rel, b_cat)
                        if travel >= min_travel_px:
                            # Actually check proximity now (heuristic check)
                            # We assume the owner is roughly where the ball is, 
                            # but this ensures no teleport-passes across court
                            if (release_f - last_emit) > debounce_f:
                                events.append({"type":"pass",
                                               "from_id": oid, "to_id": new_owner,
                                               "t_release": release_f,
                                               "t_catch": catch_f,
                                               "air_f": air,
                                               "confidence": 1.0})
                                last_emit = release_f
            i = k
        else:
            i = j + 1
    return events

def detect_dribble(owner_series, ball_centers, owner_to_center, near_px, far_px, max_gap_f=10):
    """
    Robust Dribble Detector with Merging:
    1. Bridges 'None' gaps.
    2. Detects bounces with relaxed thresholds.
    3. NEW: Merges close events (fragmentation fix).
    """
    raw_events = []
    n = len(owner_series)
    i = 0
    
    while i < n:
        f_start, oid = owner_series[i]
        
        if oid is None:
            i += 1
            continue
            
        # 1. Build a "Bridged" Run
        run_frames = []
        j = i
        while j < n:
            curr_f, curr_owner = owner_series[j]
            if curr_owner == oid:
                run_frames.append(curr_f)
                j += 1
            elif curr_owner is None:
                found_return = False
                look_ahead = 1
                while (j + look_ahead) < n and look_ahead <= max_gap_f:
                    next_f, next_o = owner_series[j + look_ahead]
                    if next_o == oid:
                        found_return = True
                        break
                    if next_o is not None and next_o != oid:
                        break 
                    look_ahead += 1
                if found_return:
                    for k in range(look_ahead):
                        run_frames.append(owner_series[j+k][0])
                    j += look_ahead
                else:
                    break
            else:
                break
        
        # 2. Analyze Run
        if run_frames:
            labels = []
            for fr in run_frames:
                bc = ball_centers.get(fr)
                pc = owner_to_center.get(fr, {}).get(oid)
                
                if bc and pc:
                    d = dist(bc, pc)
                    if   d < near_px: lbl = "near"
                    elif d > far_px:  lbl = "far"
                    else:             lbl = "mid"
                else:
                    lbl = "far"
                labels.append(lbl)

            cycles = 0
            cur = labels[0]
            t_first_push = None
            t_last_catch = None
            curr_push_start = run_frames[0] if cur == "far" else None

            for k, (fr, lbl) in enumerate(zip(run_frames[1:], labels[1:])):
                nxt = lbl if lbl != "mid" else cur
                
                if cur != "far" and nxt == "far":
                    curr_push_start = fr
                
                if cur == "far" and nxt == "near":
                    if t_first_push is None:
                        t_first_push = curr_push_start if curr_push_start else run_frames[0]
                    t_last_catch = fr
                    cycles += 1
                
                if nxt != cur:
                    cur = nxt

            if cycles >= 1 and t_first_push is not None and t_last_catch is not None:
                idx_start = run_frames.index(t_first_push)
                idx_end   = run_frames.index(t_last_catch)
                
                # Buffer slightly to capture the hand contact
                t_start_buf = run_frames[max(0, idx_start - 2)]
                t_end_buf   = run_frames[min(len(run_frames)-1, idx_end + 2)]
                
                raw_events.append({
                    "type": "dribble",
                    "player_id": oid,
                    "t_start": t_start_buf,
                    "t_end": t_end_buf,
                    "cycles": cycles
                })
        i = j

    # 3. Merge Logic (New)
    if not raw_events:
        return []
    
    # Sort by start time
    raw_events.sort(key=lambda x: x['t_start'])
    
    merged_events = []
    current_evt = raw_events[0]
    
    # Threshold for merging: 15 frames (1.5s at 10fps) covers hesitations
    MERGE_GAP = 15 
    
    for next_evt in raw_events[1:]:
        # If same player AND close in time
        if (next_evt['player_id'] == current_evt['player_id'] and 
            (next_evt['t_start'] - current_evt['t_end']) <= MERGE_GAP):
            
            # Extend current event
            current_evt['t_end'] = next_evt['t_end']
            current_evt['cycles'] += next_evt['cycles']
        else:
            merged_events.append(current_evt)
            current_evt = next_evt
    merged_events.append(current_evt)

    return merged_events


# def detect_dribble(owner_series, ball_centers, owner_to_center, near_px, far_px, max_gap_f=10):
#     """
#     Robust Dribble Detector:
#     1. Bridges 'None' gaps up to max_gap_f.
#     2. Treats missing ball as 'far'.
#     3. Trims start/end times to the actual bounce (IoU Fix).
#     """
#     events = []
#     n = len(owner_series)
#     i = 0
    
#     while i < n:
#         f_start, oid = owner_series[i]
        
#         if oid is None:
#             i += 1
#             continue
            
#         # 1. Build a "Bridged" Run
#         run_frames = []
#         j = i
#         while j < n:
#             curr_f, curr_owner = owner_series[j]
#             if curr_owner == oid:
#                 run_frames.append(curr_f)
#                 j += 1
#             elif curr_owner is None:
#                 found_return = False
#                 look_ahead = 1
#                 while (j + look_ahead) < n and look_ahead <= max_gap_f:
#                     next_f, next_o = owner_series[j + look_ahead]
#                     if next_o == oid:
#                         found_return = True
#                         break
#                     if next_o is not None and next_o != oid:
#                         break 
#                     look_ahead += 1
#                 if found_return:
#                     for k in range(look_ahead):
#                         run_frames.append(owner_series[j+k][0])
#                     j += look_ahead
#                 else:
#                     break
#             else:
#                 break
        
#         # 2. Analyze Run
#         if run_frames:
#             labels = []
#             for fr in run_frames:
#                 bc = ball_centers.get(fr)
#                 pc = owner_to_center.get(fr, {}).get(oid)
                
#                 if bc and pc:
#                     d = dist(bc, pc)
#                     if   d < near_px: lbl = "near"
#                     elif d > far_px:  lbl = "far"
#                     else:             lbl = "mid"
#                 else:
#                     lbl = "far"
#                 labels.append(lbl)

#             cycles = 0
#             cur = labels[0]
            
#             # TRIM FIX: Track exact start/end of the bouncing motion
#             t_first_push = None
#             t_last_catch = None
#             curr_push_start = run_frames[0] if cur == "far" else None

#             for k, (fr, lbl) in enumerate(zip(run_frames[1:], labels[1:])):
#                 # nxt is state at run_frames[k+1]
#                 nxt = lbl if lbl != "mid" else cur
                
#                 # Near -> Far (The Push)
#                 if cur != "far" and nxt == "far":
#                     curr_push_start = fr
                
#                 # Far -> Near (The Catch)
#                 if cur == "far" and nxt == "near":
#                     if t_first_push is None:
#                         # Start of the very first bounce
#                         t_first_push = curr_push_start if curr_push_start else run_frames[0]
                    
#                     t_last_catch = fr
#                     cycles += 1
                
#                 if nxt != cur:
#                     cur = nxt

#             if cycles >= 1 and t_first_push is not None and t_last_catch is not None:
#                 # Add a tiny buffer (2 frames) to capture the hand contact
#                 idx_start = run_frames.index(t_first_push)
#                 idx_end   = run_frames.index(t_last_catch)
                
#                 t_start_buf = run_frames[max(0, idx_start - 2)]
#                 t_end_buf   = run_frames[min(len(run_frames)-1, idx_end + 2)]
                
#                 events.append({
#                     "type": "dribble",
#                     "player_id": oid,
#                     "t_start": t_start_buf,
#                     "t_end": t_end_buf,
#                     "cycles": cycles
#                 })
        
#         i = j

#     return events
# def detect_dribble(owner_series, ball_centers, owner_to_center, near_px, far_px, max_gap_f=10):
#     """
#     Robust Dribble Detector:
#     1. Bridges 'None' gaps up to max_gap_f.
#     2. Treats missing ball as 'far'.
#     3. Counts the FIRST bounce (Near->Far->Near) immediately.
#     """
#     events = []
#     n = len(owner_series)
#     i = 0
    
#     while i < n:
#         f_start, oid = owner_series[i]
        
#         # Skip unowned
#         if oid is None:
#             i += 1
#             continue
            
#         # 1. Build a "Bridged" Run
#         run_frames = []
#         j = i
#         while j < n:
#             curr_f, curr_owner = owner_series[j]
#             if curr_owner == oid:
#                 run_frames.append(curr_f)
#                 j += 1
#             elif curr_owner is None:
#                 found_return = False
#                 look_ahead = 1
#                 while (j + look_ahead) < n and look_ahead <= max_gap_f:
#                     next_f, next_o = owner_series[j + look_ahead]
#                     if next_o == oid:
#                         found_return = True
#                         break
#                     if next_o is not None and next_o != oid:
#                         break 
#                     look_ahead += 1
#                 if found_return:
#                     for k in range(look_ahead):
#                         run_frames.append(owner_series[j+k][0])
#                     j += look_ahead
#                 else:
#                     break
#             else:
#                 break
        
#         # 2. Analyze Run
#         if run_frames:
#             labels = []
#             debug_dists = [] # Debugging
            
#             for fr in run_frames:
#                 bc = ball_centers.get(fr)
#                 pc = owner_to_center.get(fr, {}).get(oid)
                
#                 if bc and pc:
#                     d = dist(bc, pc)
#                     debug_dists.append(d)
#                     if   d < near_px: lbl = "near"
#                     elif d > far_px:  lbl = "far"
#                     else:             lbl = "mid"
#                 else:
#                     # Missing ball/player = treat as 'far' (dribbling/occluded)
#                     lbl = "far"
#                 labels.append(lbl)

#             # Debug output (you can remove this later)
#             if len(debug_dists) > 4:
#                 min_d, max_d = min(debug_dists), max(debug_dists)
#                 # Uncomment to see ranges
#                 # print(f"[DEBUG] PID {oid} frames={len(run_frames)} | Dist Range: {min_d:.1f}px - {max_d:.1f}px | Thresholds: <{near_px} >{far_px}")

#             cycles = 0
#             cur = labels[0]
#             last_flip_f = None 

#             for (fr, lbl) in zip(run_frames[1:], labels[1:]):
#                 nxt = lbl if lbl != "mid" else cur
                
#                 # Check for return to NEAR
#                 if cur == "far" and nxt == "near":
#                     if last_flip_f is None:
#                         # FIX: Count the FIRST bounce immediately
#                         cycles += 1 
#                         last_flip_f = fr
#                     else:
#                         period = fr - last_flip_f
#                         if period >= 3: 
#                             cycles += 1
#                         last_flip_f = fr
                
#                 if nxt != cur:
#                     cur = nxt

#             if cycles >= 1:
#                 events.append({"type":"dribble",
#                                "player_id": oid,
#                                "t_start": run_frames[0],
#                                "t_end": run_frames[-1],
#                                "cycles": cycles})
        
#         # Advance i
#         i = j

#     return events


# def detect_dribble(owner_series, ball_centers, owner_to_center, near_px, far_px, max_gap_f=25):
#     events = []
#     n = len(owner_series)
#     i = 0
    
#     while i < n:
#         f_start, oid = owner_series[i]
#         if oid is None:
#             i += 1
#             continue
            
#         # --- 1. Bridge Gaps (Same as before) ---
#         run_frames = []
#         j = i
#         while j < n:
#             curr_f, curr_owner = owner_series[j]
#             if curr_owner == oid:
#                 run_frames.append(curr_f)
#                 j += 1
#             elif curr_owner is None:
#                 # Look ahead
#                 found_return = False
#                 look_ahead = 1
#                 while (j + look_ahead) < n and look_ahead <= max_gap_f:
#                     next_f, next_o = owner_series[j + look_ahead]
#                     if next_o == oid:
#                         found_return = True
#                         break
#                     if next_o is not None and next_o != oid:
#                         break 
#                     look_ahead += 1
#                 if found_return:
#                     for k in range(look_ahead):
#                         run_frames.append(owner_series[j+k][0])
#                     j += look_ahead
#                 else:
#                     break
#             else:
#                 break
        
#         # --- 2. Analyze & DEBUG ---
#         if run_frames:
#             labels = []
#             # DEBUG: Track distances to see why thresholds fail
#             debug_dists = [] 
            
#             for fr in run_frames:
#                 bc = ball_centers.get(fr)
#                 pc = owner_to_center.get(fr, {}).get(oid)
                
#                 if bc and pc:
#                     d = dist(bc, pc)
#                     debug_dists.append(d)
#                     if   d < near_px: lbl = "near"
#                     elif d > far_px:  lbl = "far"
#                     else:             lbl = "mid"
#                 else:
#                     # Missing ball/player = treat as FAR (dribble extension)
#                     lbl = "far"
#                     # We don't add to debug_dists because d is undefined
#                 labels.append(lbl)

#             # PRINT DIAGNOSTICS (Remove this later)
#             if len(debug_dists) > 5:
#                 min_d = min(debug_dists)
#                 max_d = max(debug_dists)
#                 # Only print if we are somewhat confused (e.g. valid length but no event)
#                 print(f"[DEBUG] PID {oid} frames={len(run_frames)} | Dist Range: {min_d:.1f}px - {max_d:.1f}px | Thresholds: <{near_px} >{far_px}")

#             cycles = 0
#             cur = labels[0]
#             last_flip_f = None 

#             for (fr, lbl) in zip(run_frames[1:], labels[1:]):
#                 nxt = lbl if lbl != "mid" else cur
                
#                 # Check for return to NEAR
#                 if cur == "far" and nxt == "near":
#                     if last_flip_f is None:
#                         last_flip_f = fr
#                     else:
#                         period = fr - last_flip_f
#                         if period >= 3: 
#                             cycles += 1
#                         last_flip_f = fr
#                 if nxt != cur:
#                     cur = nxt

#             if cycles >= 1:
#                 events.append({"type":"dribble",
#                                "player_id": oid,
#                                "t_start": run_frames[0],
#                                "t_end": run_frames[-1],
#                                "cycles": cycles})
#         i = j

#     return events
# def detect_dribble(owner_series, ball_centers, owner_to_center, near_px, far_px, max_gap_f=10):
#     """
#     Robust Dribble Detector:
#     - Bridges 'None' gaps (up to max_gap_f) if ownership returns to the same player.
#     - Treats missing ball frames as 'far' (logical assumption: if I lost it, it's not in my hand).
#     """
#     events = []
#     n = len(owner_series)
#     i = 0
    
#     while i < n:
#         f_start, oid = owner_series[i]
        
#         # Skip unowned segments
#         if oid is None:
#             i += 1
#             continue
            
#         # 1. Build a "Bridged" Run
#         # We collect frames as long as the owner is OID *OR* it's a short None gap leading back to OID
#         run_frames = []
#         j = i
#         while j < n:
#             curr_f, curr_owner = owner_series[j]
            
#             if curr_owner == oid:
#                 run_frames.append(curr_f)
#                 j += 1
#             elif curr_owner is None:
#                 # Look ahead to see if OID returns within max_gap_f
#                 found_return = False
#                 look_ahead = 1
#                 while (j + look_ahead) < n and look_ahead <= max_gap_f:
#                     next_f, next_o = owner_series[j + look_ahead]
#                     if next_o == oid:
#                         found_return = True
#                         break
#                     if next_o is not None and next_o != oid:
#                         break # Stolen/Passed to someone else
#                     look_ahead += 1
                
#                 if found_return:
#                     # Bridge the gap! Add these None frames to the run
#                     for k in range(look_ahead):
#                         run_frames.append(owner_series[j+k][0])
#                     j += look_ahead
#                 else:
#                     # No return, end of run
#                     break
#             else:
#                 # Owner changed to someone else
#                 break
        
#         # 2. Analyze the Bridged Run for Dribbles
#         if run_frames:
#             labels = []
#             for fr in run_frames:
#                 bc = ball_centers.get(fr)
#                 pc = owner_to_center.get(fr, {}).get(oid)
                
#                 # CRITICAL: If ball is missing (bc is None), treat as 'far' (dribbling/occluded)
#                 # If ball is visible, calculate distance
#                 if bc and pc:
#                     d = dist(bc, pc)
#                     if   d < near_px: lbl = "near"
#                     elif d > far_px:  lbl = "far"
#                     else:             lbl = "mid"
#                 else:
#                     # Missing ball/player logic -> likely mid-dribble or lost track
#                     # We treat missing ball as 'far' to encourage cycle counting
#                     lbl = "far" 
#                 labels.append(lbl)

#             cycles = 0
#             cur = labels[0]
#             last_flip_f = None 

#             for (fr, lbl) in zip(run_frames[1:], labels[1:]):
#                 nxt = lbl if lbl != "mid" else cur
                
#                 # Count cycle on return to near
#                 if cur == "far" and nxt == "near":
#                     if last_flip_f is None:
#                         last_flip_f = fr
#                     else:
#                         period = fr - last_flip_f
#                         if period >= 3: 
#                             cycles += 1
#                         last_flip_f = fr
                
#                 if nxt != cur:
#                     cur = nxt

#             if cycles >= 1:
#                 events.append({"type":"dribble",
#                                "player_id": oid,
#                                "t_start": run_frames[0],
#                                "t_end": run_frames[-1],
#                                "cycles": cycles})
        
#         # Advance i
#         i = j

#     return events
# def detect_dribble(owner_series, ball_centers, owner_to_center, near_px, far_px):
#     """
#     FIX 4: Fixes initialization bug (don't start at frame 0).
#     FIX 5: Removes upper bound on period (handles hesitation) and allows faster toggles.
#     """
#     events = []
#     i, n = 0, len(owner_series)
#     while i < n:
#         f0, oid = owner_series[i]
#         j = i
#         while j+1 < n and owner_series[j+1][1] == oid:
#             j += 1
#         if oid is not None:
#             frames = [owner_series[k][0] for k in range(i, j+1)]
#             labels = []
#             for fr in frames:
#                 bc = ball_centers.get(fr)
#                 pc = owner_to_center.get(fr, {}).get(oid)
#                 d = dist(bc, pc) if (bc and pc) else 1e9
#                 if   d < near_px: lbl = "near"
#                 elif d > far_px:  lbl = "far"
#                 else:             lbl = "mid"
#                 labels.append(lbl)

#             cycles = 0
#             cur = labels[0]
#             last_flip_f = None # Fixed init

#             for (fr, lbl) in zip(frames[1:], labels[1:]):
#                 nxt = lbl if lbl != "mid" else cur
                
#                 # Count cycle on return to near
#                 if cur == "far" and nxt == "near":
#                     if last_flip_f is None:
#                         # First downstroke detected, start timer
#                         last_flip_f = fr
#                     else:
#                         period = fr - last_flip_f
#                         # Relaxed constraints: just need physical time, no upper bound
#                         if period >= 3: 
#                             cycles += 1
#                         last_flip_f = fr
                
#                 if nxt != cur:
#                     cur = nxt

#             if cycles >= 1: # Lowered threshold slightly as we fixed init
#                 events.append({"type":"dribble",
#                                "player_id": oid,
#                                "t_start": frames[0],
#                                "t_end": frames[-1],
#                                "cycles": cycles})
#         i = j + 1
#     return events

def detect_shot(owner_series, ball_centers, min_air_f, min_speed_px, rim_xy=None,
                rim_seek_f=3, rim_px=150):
    """
    FIX 6: Uses max velocity in first few frames (robust to index error).
    FIX 7: Prioritizes air+speed over rim if rim is unreliable.
    """
    events = []
    i = 0; n = len(owner_series)
    while i < n:
        f0, oid = owner_series[i]
        if oid is None:
            i += 1; continue
        j = i
        while j+1 < n and owner_series[j+1][1] == oid:
            j += 1
        release_f = owner_series[j][0]

        k = j + 1
        air = 0
        while k < n and owner_series[k][1] is None:
            air += 1; k += 1

        ok = False
        subtype = "shot"

        # (A) classic: min air + speed
        if air >= min_air_f:
            series = []
            # Look forward from release
            for fr in range(release_f, min(owner_series[-1][0], release_f+6)):
                if fr in ball_centers: series.append(ball_centers[fr])
            
            if len(series) > 2:
                # Take max speed in first few frames to avoid index error/noise
                vs = [dist(series[x], series[x-1]) for x in range(1, len(series))]
                if max(vs) >= min_speed_px:
                    ok = True

        # (B) rim-seek (only if rim_xy provided)
        if not ok and rim_xy is not None:
            looked = 0
            t = j + 1
            while t < n and looked < rim_seek_f:
                bc = ball_centers.get(owner_series[t][0])
                if bc and dist(bc, rim_xy) <= rim_px:
                    ok = True; break
                looked += 1; t += 1

        if ok:
            # Check layup dist
            if rim_xy and release_f in ball_centers and dist(ball_centers[release_f], rim_xy) < 120:
                subtype = "layup"
            events.append({"type": subtype,
                           "shooter_id": oid,
                           "t_release": release_f,
                           "air_frames": air})
        i = j + 1 + air
    return events

# --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--seq", default=None)
    ap.add_argument("--tracker", default="botsort")
    ap.add_argument("--mot_root", default="data/meta/mot")
    ap.add_argument("--res_root", default="results/track")
    ap.add_argument("--out_root", default="results/actions_auto")
    ap.add_argument("--player_cls", type=int, default=0)
    ap.add_argument("--ball_cls",   type=int, default=1)
    # tuning 
    ap.add_argument("--own_px",       type=float, default=OWN_MAX_PX)
    ap.add_argument("--own_hys",      type=float, default=OWN_HYSTERESIS)
    ap.add_argument("--near_px",      type=float, default=NEAR_PX)
    ap.add_argument("--far_px",       type=float, default=FAR_PX)
    ap.add_argument("--shot_air",     type=int,   default=SHOT_MIN_AIR_F)
    ap.add_argument("--shot_speed",   type=float, default=SHOT_MIN_SPEED)
    ap.add_argument("--pass_air_min", type=int,   default=PASS_MIN_AIR_F)
    ap.add_argument("--pass_air_max", type=int,   default=PASS_MAX_AIR_F)
    ap.add_argument("--pass_travel",  type=float, default=PASS_MIN_TRAVEL_PX)
    ap.add_argument("--debounce_f",   type=int,   default=DEBOUNCE_F)
    ap.add_argument("--min_hold_f",   type=int,   default=MIN_HOLD_F)
    ap.add_argument("--ball_smooth",  type=int,   default=BALL_SMOOTH_F)
    ap.add_argument("--team_json", default=None)
    args = ap.parse_args()

    mot_split = Path(args.mot_root)/args.split
    res_split = Path(args.res_root)/args.tracker/args.split
    out_split = Path(args.out_root)/args.split
    out_split.mkdir(parents=True, exist_ok=True)

    if args.seq:
        seq_dirs = [mot_split/args.seq]
    else:
        seq_dirs = [d for d in sorted(mot_split.iterdir()) if d.is_dir()]

    for seq_dir in seq_dirs:
        seq = seq_dir.name
        mot_file = res_split/seq/f"{seq}.txt"
        if not mot_file.exists(): continue

        W,H,L,fps = read_seqinfo(seq_dir)
        own_max = scale_by_width(W, args.own_px)
        own_hys = scale_by_width(W, args.own_hys)
        near_px = scale_by_width(W, args.near_px)
        far_px  = scale_by_width(W, args.far_px)
        pass_travel = scale_by_width(W, args.pass_travel)
        shot_speed  = scale_by_width(W, args.shot_speed)

        by_f = load_mot(mot_file)
        if not by_f: continue
        frames = sorted(by_f.keys())
        if L: frames = list(range(1, max(L, frames[-1])+1))

        players_by_f = {f:[t for t in by_f.get(f,[]) if t[1]==args.player_cls] for f in frames}
        ball_by_f    = {f:[t for t in by_f.get(f,[]) if t[1]==args.ball_cls]   for f in frames}

        ball_centers = {}
        owner_to_center = defaultdict(dict)
        for f in frames:
            if ball_by_f.get(f):
                b = max(ball_by_f[f], key=lambda z:z[-1])
                ball_centers[f] = cxcy_from_box(b)
            for p in players_by_f.get(f, []):
                owner_to_center[f][p[0]] = cxcy_from_box(p)

        ball_centers = {**ball_centers, **smooth_centers(frames, ball_centers, radius=args.ball_smooth)}

        owner_series = compute_ownership(frames, players_by_f, ball_by_f,
                                         own_max, own_hys, args.player_cls, args.ball_cls,
                                         min_hold_f=args.min_hold_f)

        passes = detect_pass(owner_series, ball_centers, near_px,
                             args.pass_air_min, args.pass_air_max, pass_travel, args.debounce_f)
        dribbles = detect_dribble(owner_series, ball_centers, owner_to_center, near_px, far_px)

        rim_xy = None
        rim_json = Path("data/meta/court")/f"{seq}.json"
        if rim_json.exists():
            try:
                rim = json.loads(rim_json.read_text()).get("rim_xy")
                if rim and len(rim)==2: rim_xy = (float(rim[0]), float(rim[1]))
            except: pass

        shots = detect_shot(owner_series, ball_centers, args.shot_air, shot_speed, rim_xy=rim_xy)
        events = sorted(passes + dribbles + shots, key=lambda e: e.get("t_release", e.get("t_start", 10**9)))

        print(f"[{seq}] frames={len(frames)} | pass={len(passes)} shot={len(shots)} drib={len(dribbles)}")
        out_f = out_split/f"{seq}.json"
        out_f.write_text(json.dumps({"sequence": seq, "fps": fps, "events": events}, indent=2))

if __name__ == "__main__":
    main()
