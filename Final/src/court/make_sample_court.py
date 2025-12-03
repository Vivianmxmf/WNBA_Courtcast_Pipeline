#!/usr/bin/env python3
import argparse
import numpy as np, cv2, math

# Court dims (feet)
L, W = 94.0, 50.0
HL, HW = L/2, W/2
LANE_W = 16.0
LANE_H = 19.0
ARC_R  = 23.75
RIM_X  = 47.0 - 4.0
CENTER_R = 6.0
FT_DIST = 15.0  # FT circle radius (NBA 6ft, but we just draw key arcs cleanly)

def ft_to_px(xf, yf, Wpx, Hpx, pad):
    scale = min((Wpx-2*pad)/L, (Hpx-2*pad)/W)
    u = Wpx/2 + xf*scale
    v = Hpx/2 - yf*scale
    return int(round(u)), int(round(v)), scale

def draw_court(Wpx, Hpx, line=4, pad=60, wood=(210,205,190)):
    img = np.full((Hpx, Wpx, 3), wood, np.uint8)
    color = (40,40,40)
    # outer rectangle
    p1 = ft_to_px(-HL, -HW, Wpx, Hpx, pad)[0:2]
    p2 = ft_to_px( HL,  HW, Wpx, Hpx, pad)[0:2]
    cv2.rectangle(img, p1, p2, color, line)

    # center line & circle
    c0 = ft_to_px(0,0,Wpx,Hpx,pad)[0:2]
    cv2.line(img, (ft_to_px(0,-HW,Wpx,Hpx,pad)[0], ft_to_px(0,-HW,Wpx,Hpx,pad)[1]),
                  (ft_to_px(0, HW,Wpx,Hpx,pad)[0], ft_to_px(0, HW,Wpx,Hpx,pad)[1]), color, line)
    _,_,s = ft_to_px(0,0,Wpx,Hpx,pad)
    cv2.circle(img, c0, int(round(CENTER_R*s)), color, line)

    # lanes (both sides)
    for side in [-1, +1]:
        base_x = side*HL
        # lane box
        x1 = base_x - side*LANE_H
        y1 = -LANE_W/2; y2 = LANE_W/2
        cv2.rectangle(img, ft_to_px(x1,y1,Wpx,Hpx,pad)[0:2],
                           ft_to_px(base_x,y2,Wpx,Hpx,pad)[0:2], color, line)
        # three-point arc (approx with circle)
        rim_x = side*(RIM_X)
        cv2.circle(img, ft_to_px(rim_x,0,Wpx,Hpx,pad)[0:2], int(round(ARC_R*s)), color, line)
        # straight 3pt sidelines (clip visually by outer rect)
        for y in [-HW, HW]:
            u1,v1,_ = ft_to_px(base_x, y, Wpx,Hpx,pad)
            u2,v2,_ = ft_to_px(base_x- side* (ARC_R- (W/2 - abs(y))), y, Wpx,Hpx,pad)
            cv2.line(img, (u2,v2), (u1,v1), color, line)

    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="result/assets/court_topdown.png")
    ap.add_argument("--W", type=int, default=1800)
    ap.add_argument("--H", type=int, default=960)
    ap.add_argument("--line", type=int, default=4)
    ap.add_argument("--pad", type=int, default=60)
    args = ap.parse_args()
    img = draw_court(args.W, args.H, line=args.line, pad=args.pad)
    cv2.imwrite(args.out, img)
    print(f"[OK] wrote {args.out}")

if __name__ == "__main__":
    main()