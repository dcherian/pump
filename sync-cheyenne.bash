rsync -ahvPi \
      --exclude 'sync-cheyenne.bash' \
      --exclude 'datasets' \
      --exclude 'movies' \
      --exclude 'glorys' \
      --exclude '__pycache__' \
      --exclude '*.zarr' \
      --exclude 'glade' \
      --exclude '.git' \
      --exclude 'pump.org' \
      chdata:~/pump/* .
rsync -ahvPi --exclude 'modis' chdata:~/pump/datasets/* datasets/
rsync -ahvPi chdata:~/pump/movies/sst_jq movies/
rsync -ahvPi ~/pump/images/* ~/pump/hugo/static/ox-hugo/
