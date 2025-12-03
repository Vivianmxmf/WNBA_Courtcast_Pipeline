CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
python3 src/yolo/train_yolov8.py   --device 0,1,2,3,4,5,6,7 --amp false

# VAL (maps to val_fixed under the hood)
python3 src/track/build_mot_from_lists.py \
  --split val \
  --frame_list data/meta/splits/val.txt \
  --gt_master data/meta/cvat_exports/val_gt.txt \
  --images_root data/dataset/images \
  --fps 2

# TRAIN
python3 src/track/build_mot_from_lists.py \
  --split train \
  --frame_list data/meta/splits/train.txt \
  --gt_master data/meta/cvat_exports/train_gt.txt \
  --images_root data/dataset/images \
  --fps 30

### For 2ps
# BoT-SORT for both classes
python3 src/track/run_tracker.py \
  --split val \
  --tracker botsort \
  --classes 0,1 \
  --conf 0.50 \
  --iou 0.60 \
  --name botsort_2ps
# ByteTrack for both classes
python3 src/track/run_tracker.py \
  --split val \
  --tracker bytetrack \
  --classes 0,1 \
  --conf 0.30 \
  --iou 0.50 \
  --name bytetrack_both
#  convert to MOT once
python3 src/track/ultra_to_mot.py \
  --ultra_root results/track \
  --tracker botsort \
  --split val \
  --mot_out results/track/botsort/val \
  --mot_gt data/meta/mot
# (or replace 'botsort' with 'bytetrack')
# awk -F, 'NF>=8 {print int($8)}' data/meta/mot/val/*/gt/gt.txt | sort -n | uniq -c

## quick sanity check again
cut -d, -f8 results/track/botsort/val/v1_p4/v1_p4.txt | sort | uniq -c
awk -F, '$8==1{print $1}' results/track/botsort/val/v1_p4/v1_p4.txt | sort -u | wc -l

# Evaluate (players are class 1, ball: class 2):
/usr/bin/python3 src/track/eval_track_internal.py \
  --mot_gt data/meta/mot/val \
  --res_root results/track/botsort/val \
  --class_id 1     

## For 10ps
rm -rf results/track
python3 src/track/run_tracker.py \
  --split val10 \
  --tracker botsort \
  --classes 0,1 \
  --conf 0.5 \
  --iou 0.60 \
  --name botsort_10fps

python3 tools/mot_make_seqinfo.py --root data/meta/mot/val10 --fps 10

python3 src/track/ultra_to_mot.py \
  --ultra_root results/track \
  --tracker botsort \
  --split val10 \
  --mot_out results/track/botsort/val10 \
  --mot_gt data/meta/mot/val10

python3 tools/mot_fix_frame_ids.py \
  --mot_dir results/track/botsort/val10 \
  --gt_root data/meta/mot/val10

# verify again
awk -F, 'max<$1{max=$1} END{print "max_frame=",max}' results/track/botsort/val10/v1_p4/v1_p4.txt

### for both 2ps and 10ps
#ID-overlay clip per sequence
python3 - <<'PY'
import cv2, csv
from pathlib import Path

split="val"; tracker="botsort"
root = Path("data/meta/mot")/split
res  = Path("results/track")/tracker/split
outd = Path("results/overlay2"); outd.mkdir(parents=True, exist_ok=True)

for seq in sorted([d.name for d in root.iterdir() if d.is_dir()]):
    mot = {}
    ftxt = res/seq/f"{seq}.txt"
    if not ftxt.exists(): continue
    for row in csv.reader(open(ftxt)):
        if len(row)<8: continue
        fr=int(float(row[0])); tid=int(float(row[1])); x,y,w,h=map(float,row[2:6]); cls=int(float(row[7]))
        mot.setdefault(fr,[]).append((tid,cls,x,y,w,h))
    imgdir = root/seq/"img1"
    imgs = sorted(imgdir.glob("*.png")) or sorted(imgdir.glob("*.jpg"))
    if not imgs: continue
    H,W,_ = cv2.imread(str(imgs[0])).shape
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    vw=cv2.VideoWriter(str(outd/f"{seq}.mp4"), fourcc, 2, (W,H))
    for fr,im in enumerate(imgs, start=1):
        frame = cv2.imread(str(im))
        for tid,cls,x,y,w,h in mot.get(fr,[]):
            color=(50,220,255) if cls==0 else (40,160,40)
            p1=(int(x),int(y)); p2=(int(x+w),int(y+h))
            cv2.rectangle(frame,p1,p2,color,2)
            cv2.putText(frame,f"ID {tid}",(p1[0],max(20,p1[1]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(frame,f"ID {tid}",(p1[0],max(20,p1[1]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv2.LINE_AA)
        vw.write(frame)
    vw.release()
    print("[OK]", seq)
PY

#best for 2 fps

python3 src/rules/rules_actions.py \
  --split val --tracker botsort \
  --player_cls 0 --ball_cls 1 \
  --own_px 143.5 --own_hys 20 --min_hold_f 1 \
  --near_px 50 --far_px 80 \
  --pass_air_min 0 --pass_air_max 4 --pass_travel 60 --debounce_f 3 \
  --shot_air 2 --shot_speed 2 --ball_smooth 3 

python3 src/rules/rules_actions.py \
  --split val --tracker botsort \
  --player_cls 0 --ball_cls 1 \
  --own_px 143.5 --own_hys 20 --min_hold_f 1 \
  --near_px 50 --far_px 80 \
  --pass_air_min 0 --pass_air_max 4 --pass_travel 60 --debounce_f 3 \
  --shot_air 2 --shot_speed 3 --ball_smooth 3 

python3 src/eval/evaluate_events.py \
  --gold_glob "data/meta/actions_gold_2/val/*.csv" \
  --auto_glob "results/actions_auto/val/*.json" \
  --tol 2 --dribble_iou 0.5 \
  --out_csv results/eval/events_val_summary.csv \
  --export_viz_dir results/eval/pass_viz_val

#best for 10 fps
python3 src/rules/rules_actions.py \
  --split val10 --tracker botsort \
  --player_cls 0 --ball_cls 1 \
  --own_px 143.5 --own_hys 20 --min_hold_f 1 \
  --near_px 20 --far_px 60 \
  --pass_air_min 0 --pass_air_max 6 --pass_travel 60 --debounce_f 3 \
  --shot_air 5 --shot_speed 3 --ball_smooth 3 

python3 src/rules/rules_actions.py \
  --split val10 --tracker botsort \
  --player_cls 0 --ball_cls 1 \
  --own_px 143.5 --own_hys 20 --min_hold_f 1 \
  --near_px 20 --far_px 60 \
  --pass_air_min 0 --pass_air_max 6 --pass_travel 60 --debounce_f 3 \
  --shot_air 5 --shot_speed 1 --ball_smooth 3 


python3 src/eval/evaluate_events.py \
  --gold_glob "data/meta/actions_gold/val/*.csv" \
  --auto_glob "results/actions_auto/val10/*.json" \
  --tol 5 --dribble_iou 0.5 \
  --out_csv results/eval/events_val10_summary.csv \
  --export_viz_dir results/eval/pass_viz_val10


## clustering methods
python3 src/team/color_cluster.py \
  --game v1 \
  --algo kmeans \
  --report_csv data/meta/team_assign/v1_margin_kmeans.csv

python3 src/team/color_cluster.py \
  --game v1 \
  --algo gmm \
  --report_csv data/meta/team_assign/v1_margin_gmm.csv

python3 src/team/color_cluster.py \
  --game v1 \
  --jersey_home data/meta/team_refs/v1/home.png \
  --jersey_away data/meta/team_refs/v1/away.png \
  --report_csv data/meta/team_assign/v1_margin_prototypes.csv


# build the list of v1 sequences from the MOT split folder
seqs=$(ls -d data/meta/mot/val10/v3_p*/ | xargs -n1 basename)

# regenerate v1 crops (players only)
for s in $seqs; do
  python3 src/team/make_crops.py --tracker botsort --split val10 \
    --seq "$s" --game_id v3 --stride 10 --pad 0.05 --player_cls 0;
done


# default paths (will look for {game}_margin_kmeans.csv, etc.)
python3 src/team/inspect_team_clusters.py --game v1

python3 src/team/inspect_team_clusters.py \
  --game v1 \
  --kmeans_csv data/meta/team_assign/v1_margin_kmeans.csv \
  --gmm_csv    data/meta/team_assign/v1_margin_gmm.csv \
  --proto_csv  data/meta/team_assign/v1_margin_prototypes.csv \
  --out_dir    data/meta/team_assign/plots_v1 \
  --top_k 40 --ambig_percent 30

## Make 2D court image
python3 src/court/make_sample_court.py --out results/court_topdown.png --W 1800 --H 960 --line 4

## Compute H for each game from your two CVAT images (left/right)
python3 src/court/estimate_homography.py \
  --xml data/meta/cvat_exports/homography/game2.xml \
  --game_id v2 \
  --out_dir data/meta/homography \
  --seg_halves 1:left 2:right  
# (repeat for game2, game3)

# Visual QA of H
python3 src/court/overlay_check.py \
  --image data/dataset/images/val_fixed/v1_p7_00005.png \
  --H data/meta/homography/game1_seg1.json \
  --out overlay_game1_seg1.png


# Project tracks (homography already set up for v1/v2/v3)
python3 src/court/project_tracks.py \
  --split val\
  --tracker botsort \
  --game_id v1 \
  --team_algo kmeans \
  --crops_root data/meta/team_crops

python3 src/court/project_tracks.py \
  --split val\
  --tracker botsort \
  --game_id v2 \
  --team_algo kmeans \
  --crops_root data/meta/team_crops

python3 src/court/project_tracks.py \
  --split val\
  --tracker botsort \
  --game_id v3 \
  --team_algo kmeans \
  --crops_root data/meta/team_crops
# Prototypes
python3 src/court/project_tracks.py \
  --tracker botsort \
  --game_id v1 \
  --jersey_home data/meta/team_refs/v1/home.png \
  --jersey_away data/meta/team_refs/v1/away.png

# Visual quicklook (optional)
python3 src/court/quick_plot_tracks.py \
  --csv results/court_tracks/val10/v2_p5.csv \
  --out results/court_tracks/val10/v2_p5_preview.png \
  --title "v2_p5 (val10)"

python3 src/court/quick_plot_tracks.py \
  --csv results/court_tracks/val10/v2_p5.csv \
  --team_json data/meta/team_assign/v2.json \
  --out results/court_tracks/val10/v2_p5_preview_team.png \
  --title "v2_p5 (val10) — team colors"


# True positives of pass event
python3 src/court/visualize_on_court.py \
  --court_png results/court_topdown.png \
  --csv_glob "results/court_tracks/val10/v1_p4.csv" \
  --plot passes \
  --events_glob "results/eval/pass_viz_val10/v1_p4_pass_pairs.json" \
  --out results/vis/val10/v1_p4_pass.png --pad 60 --alpha 1.0

# Players heatmap
python3 src/court/visualize_on_court.py \
  --court_png results/court_topdown.png \
  --csv_glob "results/court_tracks/val/v1_p*.csv" \
  --plot heatmap --Class player --sigma 24  --pad 60 \
  --out results/vis/val/v1_players_heat.png

  python3 src/court/visualize_on_court.py \
  --court_png results/court_topdown.png \
  --csv_glob "results/court_tracks/val/v1_p*.csv" \
  --plot heatmap --Class ball --sigma 24  --pad 60 \
  --out results/vis/val/v1_ball_heat.png

# Pass arrows
python3 src/court/visualize_on_court.py \
  --court_png results/court_topdown.png \
  --csv_glob "results/court_tracks/val/v1_p*.csv" \
  --events_glob "results/actions_auto/val/v1_p*.json" \
  --plot passes --pad 60 \
  --out results/vis/val/v1_passes.png

python3 src/court/visualize_on_court.py \
  --court_png results/court_topdown.png \
  --csv_glob "results/court_tracks/val10/v1_p4.csv" \
  --plot shots \
  --events_glob "results/actions_auto/val10/v1_p4.json" \
  --out results/vis/v1_p4_shots.png \
  --pad 60 --alpha 1.0

python3 src/court/visualize_on_court.py \
  --court_png results/court_topdown.png \
  --csv_glob "results/court_tracks/val10/v1_p7.csv" \
  --plot dribbles \
  --events_glob "results/actions_auto/val10/v1_p7.json" \
  --out results/vis/v1_p7_dribbles.png \
  --pad 60 --alpha 1.0

python3 src/court/visualize_on_court.py \
  --court_png results/court_topdown.png \
  --csv_glob "results/court_tracks/val10/v1_p8.csv" \
  --plot all \
  --events_glob "results/actions_auto/val10/v1_p8.json" \
  --out results/vis/v1_p8_all_events.png \
  --pad 60 --alpha 1.0

