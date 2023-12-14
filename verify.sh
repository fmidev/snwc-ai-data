set -exu

if test $# -lt 3; then
  echo "usage: $0 year month day [ path ]"
  exit 1
fi

b=$ROUTINES_DATA/meps_archive/

if test $# -eq 4; then
  b=$4
fi

verify(){
  y=$1
  m=$2
  d=$3
  p=$4/$y/$m/$d

  dir=$p # ROUTINES_DATA/meps_archive/$y/$m/$d

  count=$(ls -l $dir/2*.grib2 | wc -l)

  REQUIRED_FILE_COUNT=39
  if test $count -ne $REQUIRED_FILE_COUNT; then
    echo "invalid number of files: $count (should be $REQUIRED_FILE_COUNT)"
    exit 1
  fi

  for l in 300 500 700 850 925 1000; do
    count=$(ls -l $dir/2*_isobaricInhPa_${l}*.grib2 | wc -l)
    if test $count -ne 5; then
      echo "invalid number of pressure level files for level $l: $count (should be 5)"
      exit 1
    fi
  done

  for f in $(ls -1 $dir/$y$m${d}*.grib2); do

    count=$(grib_count $f)

    if test $count -ne 24; then
      echo "file $f has invalid number of messages: $count (should be 24)"
      exit 1
    fi

    date=$(grib_get -p dataDate $f | uniq)
    count=$(echo $date | wc -l)

    if test $count -ne 1; then
      echo "file $f has more than 1 dataDates: $date"
      exit 1
    fi

    if test $date -ne $y$m$d; then
      echo "file $f has invalid dataDate: $date"
      exit 1
    fi

    dtime=$(grib_get -p dataTime $f | paste -sd+ - | bc)

    if test $dtime -ne 27600; then
      echo "data times sum does not match: $dtime, should be 27600"
      exit 1
    fi

    missing=$(grib_get -p numberOfMissing $f | paste -sd+ - | bc)

    if test $missing -ne 0; then
      echo "too many missing values: $missing, should be 0"
      exit 1
    fi

    minimum=$(grib_get -p minimum $f | awk 'NR == 1 || $1 < min { line = $0; min = $1}END{print line}')

    if awk "BEGIN {exit !($minimum < -4500)}"; then
      echo "too small minimum value: $minimum"
      exit 1
    fi

    maximum=$(grib_get -p maximum $f | awk 'NR == 1 || $1 > max { line = $0; max = $1}END{print line}')

    if awk "BEGIN {exit !($maximum > 5e5)}"; then
      echo "too large maximum value: $maximum"
      exit 1
    fi


  done

}

verify $1 $2 $3 $4

echo "All good!"
