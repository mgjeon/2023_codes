#! /bin/bash
# $1: noaanum (int)
# python series.py $1 --fname "/mnt/obsdata/isee_nlfff_v1.2/${1}" --ext "*.nc" --result_dir "/mnt/userdata/jeon_mg/series/result/${1}" 
python series.py $1 --fname "/mnt/obsdata/isee_nlfff_v1.2/${1}" --ext "*.nc" --result_dir "/userhome/jeon_mg/workspace/series_result/${1}" 