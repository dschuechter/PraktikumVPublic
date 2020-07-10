#!/bin/bash
python3 Peak-to-Total.py Cs H &
python3 Peak-to-Total.py Cs S &
python3 Peak-to-Total.py Co H &
python3 Peak-to-Total.py Co S &
wait
python3 FWHM.py Cs H &
python3 FWHM.py Co H &
python3 FWHM.py Eu H &
python3 FWHM.py Cs S &
python3 FWHM.py Co S &
python3 FWHM.py Eu S &
wait
python3 energiekalibrierung_H.py &
python3 energiekalibrierung_S.py &
python3 intrinsische_Halbwertsbreite.py Cs S &
python3 intrinsische_Halbwertsbreite.py Cs H &
python3 FWHM-Energie-Diagram.py FWHM_H_new.txt &
python3 FWHM-Energie-Diagram.py FWHM_S_new.txt &
wait
cd Messung_12h
python3 N\(E\)_getrennt.py &
python3 Cs.py 
