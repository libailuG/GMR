'''

python scripts/bvh_to_robot.py \
  --bvh_file lafan1/walk1_subject1.bvh \
  --robot unitree_g1 \
  --rate_limit

生成pkl
python scripts/bvh_to_robot.py \
  --bvh_file lafan1/walk1_subject1.bvh \
  --robot unitree_g1 \
  --save_path outputs/g1_walk.pkl

  

回放pkl
python scripts/vis_robot_motion.py \
  --robot unitree_g1 \
  --robot_motion_path outputs/g1_walk.pkl

生成csv
python scripts/batch_gmr_pkl_to_csv.py \
  --folder outputs


python scripts/tutorials/visualize_human_taixi_a2_mapping.py \
  --no-follow-camera \
  --play


先生成完整 A2 PKL
python scripts/bvh_to_robot.py \
  --bvh_file lafan1/walk1_subject1.bvh \
  --format lafan1 \
  --robot taixi_a2 \
  --motion_fps 30 \
  --save_path outputs/taixi_a2_walk1_subject1.pkl



python scripts/tutorials/select_gmr_pkl_segment.py \
  outputs/taixi_a2_walk1_subject1.pkl \
  --robot taixi_a2
'''