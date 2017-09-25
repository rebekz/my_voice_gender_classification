#Identifying Gender Based on Voice in Telephone Recording Using Machine Learning

Based on https://github.com/primaryobjects/voice-gender/

##Requirements:
- R
- Docker
- ffmpeg

##Instructions

* Process from acoustic_parametes.csv

```
Rscript process_sound.R --audio FALSE
```

* Process from sound files

```
Rscript process_sound.R --audio TRUE
```