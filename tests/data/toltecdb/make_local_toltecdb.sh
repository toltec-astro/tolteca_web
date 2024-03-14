
# this script generate a local test database for toltec db

filename=toltecdb.sqlite
mysql2sqlite -f ${filename} -d toltec -u tolteca --mysql-password tolteca \
    -h 127.0.0.1 -P 3307 \
    -t toltec obstype master -K
sqlite3 ${filename} "DELETE FROM toltec WHERE toltec.date < DATETIME('now', '-30 day');VACUUM;"
