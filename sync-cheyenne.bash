rsync -ahvPi --exclude 'glade' --exclude '.git' --exclude 'pump.org' chdata:~/pump/* .
rsync -ahvPi ~/pump/images/* ~/pump/hugo/static/ox-hugo/
