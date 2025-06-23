modal run modal_app.py \
  --cub-root /mnt/data/Flowers_converted \
  --steps 5 \
  --batch-size 35 \
  --report-interval 5 \
  --eval-batch-size 64

modal run modal_app_cub.py \
  --cub-root /mnt/data/CUB_200_2011 \
  --steps 2 \
  --batch-size 4 \
  --report-interval 2 \
  --eval-batch-size 4