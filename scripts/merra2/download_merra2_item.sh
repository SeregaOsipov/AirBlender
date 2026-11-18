#!/usr/bin/env bash
# Creates per-collection subdirectories and downloads daily MERRA-2 .nc4 files via wget.

set -u

dataset="$1"   # e.g. inst3_3d_asm_Nv
host="${2%/}"  # e.g. https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NVASM.5.12.4
inputDate="$3" # YYYY-MM-DD

year=$(date -d "$inputDate" +"%Y")
month=$(date -d "$inputDate" +"%m")
day=$(date -d "$inputDate" +"%d")
mdate=$(date -d "$inputDate" +"%Y%m%d")

# Version depends on calendar period:
# https://daac.gsfc.nasa.gov/information/documents?title=Records%20of%20MERRA-2%20Data%20Reprocessing%20and%20Service%20Changes
if [ "${year}" -eq 2021 ] && [ "${month}" -ge 6 ] && [ "${month}" -le 9 ]; then
  Vv=401
elif [ "${year}" -ge 2011 ]; then
  Vv=400
elif [ "${year}" -ge 2001 ]; then
  Vv=300
elif [ "${year}" -ge 1992 ]; then
  Vv=200
else
  Vv=100
fi

echo "input arguments are:"
echo "dataset is $dataset"
echo "year is $year"
echo "month is $month"
echo "day is $day"
echo "mdate is $mdate"
echo "host is $host"
echo "stream version is $Vv"

if [[ "$dataset" == tavgM_* ]]; then
  filename="MERRA2_${Vv}.${dataset}.${year}${month}.nc4"
  url="${host}/${year}/${filename}"
else
  filename="MERRA2_${Vv}.${dataset}.${mdate}.nc4"
  url="${host}/${year}/${month}/${filename}"
fi

echo "downloading the following url:"
echo "$url"

mkdir -p "$dataset"

wget --directory-prefix="$dataset" --no-verbose -c \
  --load-cookies ~/.urs_cookies \
  --save-cookies ~/.urs_cookies \
  --keep-session-cookies \
  --content-disposition "$url"

exit $?
