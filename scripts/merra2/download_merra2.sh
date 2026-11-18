#!/usr/bin/env bash
# Downloads a set of NASA MERRA-2 collections over [startDate, startDate + nDays] via wget.
# Recommended Python CLI alternative: airblender fetch-merra2 --start_date YYYY-MM-DD --ndays N
#
# Usage:
#   cd /path/to/MERRA2
#   ${AIRBLENDER_DIR}/scripts/merra2/download_merra2.sh 2021-01-01 365 &> log.merra2_download.2021
#
# Invariant (constant) fields:
#   https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/M2C0NXASM.5.12.4/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# WRF-Chem preset collections
datasets=(inst3_3d_asm_Nv inst1_2d_asm_Nx tavg1_2d_slv_Nx tavg1_2d_ocn_Nx tavg1_2d_lnd_Nx inst3_3d_aer_Nv inst3_3d_chm_Nv)
hosts=(
  "https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NVASM.5.12.4/"
  "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I1NXASM.5.12.4/"
  "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXSLV.5.12.4/"
  "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXOCN.5.12.4/"
  "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXLND.5.12.4/"
  "https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NVAER.5.12.4/"
  "https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NVCHM.5.12.4/"
)

# Zittis AOT partitioning preset (also available via `airblender fetch-merra2 --preset zittis_aot`):
# datasets=(tavg1_2d_aer_Nx tavgM_2d_aer_Nx inst3_3d_aer_Nv tavg3_3d_nav_Ne tavg1_2d_adg_Nx)
# hosts=(
#   "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXAER.5.12.4/"
#   "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/M2TMNXAER.5.12.4/"
#   "https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NVAER.5.12.4/"
#   "https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T3NENAV.5.12.4"
#   "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXADG.5.12.4/"
# )

startDate="$1"  # YYYY-MM-DD
nDays="$2"      # number of days after startDate (e.g. 365 for a full year)

echo "Starting at ${startDate} and will run for ${nDays} days."

downloader="${SCRIPT_DIR}/download_merra2_item.sh"
echo "the datasets list to download is ${datasets[*]}"
echo "script startDate is ${startDate}"

len=${#datasets[@]}
for ((index = 0; index < len; index++)); do
  echo "dataset index is $index"
  dataset=${datasets[$index]}
  echo "processing dataset: ${dataset}"
  host=${hosts[$index]}
  echo "host is $host"

  logsFolder="logs/${dataset}"
  mkdir -p "${logsFolder}"

  N=5  # OpenDAP/GES DISC concurrency cap
  for i in $(seq 0 "$nDays"); do
    inputDate=$(date -d "$startDate $i days" +"%Y-%m-%d")
    echo "processing date $inputDate"

    ((i = i % N)); ((i++ == 0)) && wait
    {
      "${downloader}" "$dataset" "$host" "$inputDate" > "${logsFolder}/${inputDate}.log"
      retval=$?
      # 13 indicates file exists without control file
      if [ $retval -gt 0 ] && [ $retval -ne 13 ]; then
        echo "ERROR occurred downloading the file, the code is: $retval"
        echo "use this command to try again manually:"
        echo "${downloader} $dataset $host $inputDate > ${logsFolder}/${inputDate}.log"
      fi
    } &
  done
  wait
  echo "DONE dataset $dataset"
done

echo "DONE entire download"
