# Versuch 515: Driftkammern
In dieser Repository werden die verwendeten Datensätze und Programme zur Auswertung des Versuchs 515 angeführt.
Die Datensätze dürfen nicht für eigene Protokolle verwendet werden, dienen jedoch als Orientierung. Die Programme sollen
eine schnelle Auswertung ermöglichen.

## Verwendung
### Allgemeiner Hinweis
Alle Dateien müssen mit in den Ordner mit den Datensätzen kopiert oder die Dateipfade in den einzelnen Scripten angepasst werden!
Wird das nicht gemacht, funktionieren die Scripte nicht richtig.

### Aufbereitung der Datensätze
Potentiell müssen Header vor dem ausführen der Programme entfernt werden

### Vorhandenheit aller Programme
Aufgrund von urheberrechtlichen Gründen können wir nicht das in C-Vorzufindene Programm öffentlich anbringen.

### ACHTUNG bei der Berechnung mit zusatz.py
Das Skript zusatz.py benötigt sehr lange Zeit um berechnet zu werden und ist nicht auf multicores optimiert. Demnach lohnt es sich für jeden Core eine Instanz mit verschiedenen Datenbereichen auszuwählen und anschließend die Ergebnisse zu mergen (z.B. cat *.txt -> destination).
