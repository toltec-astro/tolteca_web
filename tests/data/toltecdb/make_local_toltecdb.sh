
# this script generate a local test database for toltec db
if [[ ! $1 ]]; then
    echo "Usage: $0 <mysql_port>"
    exit
fi
port=$1
filename=toltecdb.sqlite
mysql2sqlite -f ${filename} -d toltec -u tolteca --mysql-password tolteca \
    -h 127.0.0.1 -P ${port} \
    -t toltec obstype master -K
sqlite3 ${filename} "DELETE FROM toltec WHERE toltec.date < DATETIME('now', '-30 day');VACUUM;"
