# NASA MERRA-2 Downloader Scripts

To download the entire set of MERRA-2 collections for WRF-Chem use:

### Recommended (`airblender fetch-merra2`)

```bash
airblender fetch-merra2 --start_date 2021-01-01 --ndays 365 --outdir . --preset wrfchem
# Or for Zittis AOT partitioning collections:
airblender fetch-merra2 --start_date 2021-01-01 --ndays 365 --outdir . --preset zittis_aot
```

### Legacy Shell Scripts

```bash
./download_merra2.sh 2021-01-01 365 >& log.merra2_download &
```

* Check `logs/<collection>/*.log` and `log.merra2_download` for `"ERROR"` to confirm every file downloaded cleanly.
* To download an individual file, run `./download_merra2_item.sh <collection> <host_url> <YYYY-MM-DD>`.
* Download the time-invariant `const_2d_asm_Nx` file separately from `https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/M2C0NXASM.5.12.4/`.
